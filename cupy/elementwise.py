import string

import numpy
import six

import cupy
from cupy import carray
from cupy import cuda
from cupy import internal


def _get_allocator(in_arg):
    for arg in in_arg:
        if isinstance(arg, cupy.ndarray):
            return arg.allocator
    else:
        return cuda.alloc

_typenames = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}


def _get_typename(dtype):
    global _typenames
    return _typenames[numpy.dtype(dtype)]


def _get_kernel_params_args(param_names, args):
    params = []
    kernel_args = []
    for i, x in zip(param_names, args):
        if isinstance(x, cupy.ndarray):
            t = 'CArray<{0}, {1}>'.format(_get_typename(x.dtype), x.ndim)
        elif numpy.isscalar(x):
            if not isinstance(x, numpy.generic):
                x = numpy.dtype(type(x)).type(x)
            t = _get_typename(x.dtype)
        else:
            raise ValueError()
        kernel_args.append(x)
        params.append('{} {}'.format(t, i))
    return (', '.join(params), kernel_args)


@cuda.memoize
def _get_elementwise_kernel(
        params, operation, name='kernel', options=[], preamble='',
        loop_prep='', after_loop=''):
    module_code = string.Template('''
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      CUPY_FOR(i, n) {
        ${operation};
      }
      ${after_loop};
    }
    ''').substitute(
        params=params,
        operation=operation,
        name=name,
        preamble=preamble,
        loop_prep=loop_prep,
        after_loop=after_loop)
    module = carray.compile_with_cache(module_code, options)
    return module.get_function(name)


class ElementwiseKernel(object):

    def __init__(self, param_names, operation,
                 name='kernel', options=[], **kwargs):
        self.param_names = param_names + ['n']
        self.operation = operation
        self.name = name
        self.options = []
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        n = kwargs.pop('size', None)
        reduce_dims = kwargs.pop('reduce_dims', True)
        if n is None:
            for i in args:
                if isinstance(i, cupy.ndarray):
                    n = i.size
                    break
        assert n is not None

        args = list(args)
        for i, x in enumerate(args):
            if isinstance(x, cupy.ndarray):
                if reduce_dims:
                    args[i] = x.reduced_view()
        internal.check_args_device(args)
        args.append(numpy.int32(n))
        params, kernel_args = _get_kernel_params_args(self.param_names, args)
        kernel = _get_elementwise_kernel(params, self.operation, self.name,
                                         **self.kwargs)
        kernel.linear_launch(n, kernel_args)


@cuda.memoize
def _get_ufunc_kernel(in_types, out_types, raw_out_types, is_ndarray, routine,
                      params, name, preamble):
    op = []
    for i, x in enumerate(in_types):
        a = '[i]' if is_ndarray[i] else ''
        op.append('''
                  typedef {dtype} in{i}_type;
                  in{i}_type in{i} = _x{i}{a};
                  '''.format(dtype=_get_typename(x), i=i, a=a))

    for i, x in enumerate(out_types):
        op.append('''
                  typedef {dtype} out{i}_type;
                  {raw_dtype} &out{i} = _x{j}[i];
                  '''.format(dtype=_get_typename(x),
                             raw_dtype=_get_typename(raw_out_types[i]),
                             i=i, j=i + len(in_types)))

    op.append(routine)
    operation = '\n'.join(op)
    return _get_elementwise_kernel(
        params, operation, name, preamble=preamble)


class ufunc(object):

    """Universal function.

    Attributes:
        name (str): The name of the universal function.
        nin (int): Number of input arguments.
        nout (int): Number of output arguments.
        nargs (int): Number of all arguments.

    """
    def __init__(self, name, nin, nout, ops, preamble='', doc=''):
        self.name = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        self._preamble = preamble
        self._params = ['_x{}'.format(i)
                        for i in six.moves.range(self.nargs)] + ['n']
        self.__doc__ = doc

    def __repr__(self):
        return "<ufunc '%s'>" % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for in_types, out_types, _ in self._ops:
            in_str = ''.join(numpy.dtype(t).char for t in in_types)
            out_str = ''.join(numpy.dtype(t).char for t in out_types)
            types.append('{}->{}'.format(in_str, out_str))
        return types

    def __call__(self, *args, **kwargs):
        """Applies the universal function to arguments elementwise.

        Args:
            args: Input arguments. Each of them can be a cupy.ndarray object or
                a scalar. The output arguments can be omitted or be specified
                by the ``out`` argument.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.
            dtype: Data type specifier.
            allocator (function): CuPy memory allocator. The allocator of the
                first cupy.ndarray argument is used by default.

        Returns:
            Output array or a tuple of output arrays.

        """
        if not (len(args) == self.nin or len(args) == self.nargs):
            raise TypeError('Wrong number of arguments for %s' % self.name)
        internal.check_args_device(args)

        brod = cupy.broadcast(*args)
        in_args = brod.values[:self.nin]
        assert all(i is not None for i in in_args)
        out_args = brod.values[self.nin:]
        out = kwargs.get('out', None)
        if out is not None:
            assert len(out_args) == 0
            internal.check_args_device((out,))
            out_args = [out]

        dtype = kwargs.get('dtype', None)
        allocator = kwargs.get('allocator', None)
        if allocator is None:
            allocator = _get_allocator(in_args)
        in_types, out_types, routine = self._guess_routine(in_args, dtype)

        if len(out_args) == self.nout:
            for i in out_args:
                assert isinstance(i, cupy.ndarray)
                assert i.shape == brod.shape
        else:
            out_args = [
                cupy.empty(shape=brod.shape, dtype=t, allocator=allocator)
                for t in out_types]

        if len(out_args) == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in brod.shape:
            return ret

        # TODO(okuta): reorder dimension

        inout_args = in_args + out_args + [brod.size]
        for i, x in enumerate(inout_args):
            if isinstance(x, cupy.ndarray):
                inout_args[i] = x.reduced_view()
            elif i < len(in_args):
                inout_args[i] = in_types[i](x)

        params, kernel_args = _get_kernel_params_args(self._params, inout_args)
        is_ndarray = [isinstance(x, cupy.ndarray) for x in in_args]
        raw_out_types = [x.dtype for x in out_args]

        kern = _get_ufunc_kernel(
            tuple(in_types), tuple(out_types), tuple(raw_out_types),
            tuple(is_ndarray), routine, params, self.name, self._preamble)

        kern.linear_launch(brod.size, kernel_args)
        return ret

    def _guess_routine(self, in_args, dtype):
        if dtype is None:
            for in_types, out_types, routine in self._ops:
                if all(numpy.can_cast(in_arg, in_type)
                       for in_arg, in_type in zip(in_args, in_types)):
                    return in_types, out_types, routine
        else:
            for in_types, out_types, routine in self._ops:
                if all(numpy.dtype(t) == dtype for t in out_types):
                    return in_types, out_types, routine
        raise TypeError('Wrong type of arguments for %s' % self.name)


def create_ufunc(name, ops, routine=None, preamble='', doc=''):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = [numpy.dtype(t).type for t in in_types]
        out_types = [numpy.dtype(t).type for t in out_types]
        _ops.append((in_types, out_types, rt))

    return ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble, doc)


_id = 'out0 = in0'

copy = create_ufunc(
    'cupy_copy',
    ['?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    _id)


copy_where = create_ufunc(
    'cupy_copy_where',
    ['??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d'],
    'if (in1) out0 = in0')


_divmod = create_ufunc(
    'cupy_divmod',
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'],
    'out0_type a = _floor_divide(in0, in1); out0 = a; out1 = in0 - a * in1')

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


def _get_ndarray_dtype(args):
    return tuple(a.dtype if isinstance(a, cupy.ndarray) else None
                 for a in args)
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
    if dtype is None:
        raise ValueError('dtype is None')
    return _typenames[numpy.dtype(dtype)]


def _get_kernel_params_args(param_names, args):
    params = []
    kernel_args = []
    for p, a in zip(param_names, args):
        if isinstance(a, cupy.ndarray):
            t = 'CArray<{0}, {1}>'.format(_get_typename(a.dtype), a.ndim)
        elif numpy.isscalar(a):
            if not isinstance(a, numpy.generic):
                a = numpy.dtype(type(a)).type(a)
            t = _get_typename(a.dtype)
        else:
            raise ValueError('{}'.format((p, type(a))))
        kernel_args.append(a)
        params.append('{} {}'.format(t, p))
    return (', '.join(params), kernel_args)


class ParameterInfo(object):

    def __init__(self, str, const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.const = const
        s = tuple(i for i in str.split() if len(i) != 0)
        if len(s) < 2:
            raise Exception('Syntax error: %s' % str)

        t, self.name = s[-2:]
        if len(t) == 1:
            self.ctype = t
        else:
            self.dtype = numpy.dtype(t)
            if self.dtype.name != t:
                raise ValueError('Wrong type %s' % t)
            self.ctype = _get_typename(self.dtype)

        for i in s[:-2]:
            if i == 'raw':
                self.raw = True
            else:
                raise Exception('Unknown keyward "%s"' % i)


def _get_param_info_tuple(s, const=False):
    if len(s) == 0:
        return ()
    return tuple(ParameterInfo(i, const) for i in s.strip().split(','))


@cuda.memoize
def _decide_params_type(in_params, out_params, in_args_dtype, out_args_dtype):
    type_dict = {}
    if out_args_dtype:
        assert len(out_params) == len(out_args_dtype)
        for p, a in zip(out_params, out_args_dtype):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if a != p.dtype:
                    raise TypeError(
                        'Type is mismatched %s', (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if t != a:
                    raise TypeError(
                        'Type is mismatched %s', (p.name, a, t))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_args_dtype)
    unknown_ctype = []
    for p, a in zip(in_params, in_args_dtype):
        if a is None:
            if p.dtype is None:
                unknown_ctype.append(p.ctype)
        else:
            if p.dtype is not None:
                if a != p.dtype:
                    raise TypeError(
                        'Type is mismatched %s', (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if t != a:
                    raise TypeError(
                        'Type is mismatched %s', (p.name, a, t))
            else:
                type_dict[p.ctype] = a

    in_types = tuple(p.dtype if p.dtype is not None else type_dict[p.ctype]
                     for p in in_params)
    out_types = tuple(p.dtype if p.dtype is not None else type_dict[p.ctype]
                      for p in out_params)
    return in_types, out_types, tuple(type_dict.items())


@cuda.memoize
def _get_simple_elementwise_kernel(
        params, operation, name='kernel', options=(), preamble='',
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


@cuda.memoize
def _get_elementwise_kernel(
        params, is_ndarray, types,
        kernel_params, operation, name, options=(),
        preamble='', **kwargs):
    types_preamble = '\n'.join(
        'typedef {} {};'.format(_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for x, f in zip(params, is_ndarray):
        if not f or x.raw:
            continue
        fmt = '{t} &{n} = _raw_{n}[i];'
        if x.const:
            fmt = 'const {t} {n} = _raw_{n}[i];'
        op.append(fmt.format(t=x.ctype, n=x.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        options=options, preamble=preamble, **kwargs)


def _broadcast(args, params, size_error=True):
    brod = cupy.broadcast(
        *[None if p.raw else a for p, a in zip(params, args)])
    if size_error and all(not isinstance(i, cupy.ndarray)
                          for i in brod.values):
        raise ValueError('Loop size is Undecided')
    return brod, [b if a is None else a for a, b in zip(brod.values, args)]


def _get_out_args(in_args, out_args, out_types,
                  allocator, out_shape, out_params=None):
    if len(out_args) == 0:
        if out_params is not None and any(p.raw for p in out_params):
            raise ValueError('Output array size is Undecided')
        if allocator is None:
            allocator = _get_allocator(in_args)
        out_args = [cupy.empty(shape=out_shape, dtype=t, allocator=allocator)
                    for t in out_types]
    else:
        assert len(out_args) == len(out_types)
        for i, a in enumerate(out_args):
            if not isinstance(a, cupy.ndarray):
                raise TypeError(
                    'Output arguments type must be cupy.ndarray')
            if a.shape != out_shape:
                if out_params is None or not out_params[i].raw:
                    raise ValueError('Out shape is mismatched')

    return out_args


class ElementwiseKernel(object):

    """User-defined elementwise kernel.

    This class can be used to define an elementwise kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`ElementwiseKernel.__call__` method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        operation (str): The body in the loop written in CUDA-C/C++.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        options (list): Options passed to the nvcc command.
        reduce_dims (bool): If False, the shapes of array arguments are
            kept within the kernel invocation. The shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            ndims) by default. It may make the kernel fast by reducing the
            index calculations.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """
    def __init__(self, in_params, out_params, operation,
                 name='kernel', options=(), reduce_dims=True, **kwargs):
        self.in_params = _get_param_info_tuple(in_params, True)
        self.out_params = _get_param_info_tuple(out_params)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.params = \
            self.in_params + self.out_params + _get_param_info_tuple('int32 n')
        self.operation = operation
        self.name = name
        self.options = options
        self.reduce_dims = reduce_dims
        self.kwargs = kwargs
        names = [p.name for p in self.in_params + self.out_params]
        if 'n' in names or 'i' in names:
            raise ValueError("Can not use 'i' and 'n' as a parameter name")

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or ndims are not compatible. It
        means that single ElementwiseKernel object may be compiled into
        multiple kernel binaries.

        Args:
            args: Argumens of the kernel.
            size (int): Range size of the indices. If specified, the variable
                ``n`` is set to this value. Otherwise, the result of
                broadcasting is used to determine the value of ``n``.

        """

        n = kwargs.pop('size', None)
        allocator = kwargs.get('allocator', None)

        if not (len(args) == self.nin or
                len(args) == self.nin + self.nout):
            raise TypeError('Wrong number of arguments for %s' % self.name)
        assert all(i is not None for i in args)
        internal.check_args_device(args)

        brod, value = _broadcast(args, self.params, n is None)
        in_args = value[:self.nin]
        out_args = value[self.nin:]
        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            _get_ndarray_dtype(in_args), _get_ndarray_dtype(out_args))

        out_args = _get_out_args(
            in_args, out_args, out_types, allocator, brod.shape,
            self.out_params)

        if len(out_args) == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if n is None:
            n = brod.size
        if n == 0:
            return ret

        inout_args = in_args + out_args + [numpy.int32(n)]
        param_names = []
        for i, x in enumerate(inout_args):
            name = self.params[i].name
            if isinstance(x, cupy.ndarray):
                if not self.params[i].raw:
                    name = "_raw_" + name
                if self.reduce_dims:
                    inout_args[i] = x.reduced_view()
            elif i < len(in_args):
                inout_args[i] = in_types[i].type(x)
            param_names.append(name)

        kernel_params, kernel_args = _get_kernel_params_args(param_names,
                                                             inout_args)
        is_ndarray = tuple(isinstance(x, cupy.ndarray) for x in inout_args)
        kern = _get_elementwise_kernel(
            self.params, is_ndarray, types,
            kernel_params, self.operation, self.name, self.options,
            **self.kwargs)
        kern.linear_launch(n, kernel_args)
        return ret


@cuda.memoize
def _get_ufunc_kernel(in_types, out_types, raw_out_types, is_ndarray, params,
                      routine, name, preamble):
    op = []
    for i, x in enumerate(in_types):
        a = '[i]' if is_ndarray[i] else ''
        op.append('''
                  typedef {dtype} in{i}_type;
                  const in{i}_type in{i} = _x{i}{a};
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
    return _get_simple_elementwise_kernel(
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
            in_str = ''.join(t.char for t in in_types)
            out_str = ''.join(t.char for t in out_types)
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
        out = kwargs.get('out', None)
        dtype = kwargs.get('dtype', None)
        allocator = kwargs.get('allocator', None)

        if not (len(args) == self.nin or len(args) == self.nargs):
            raise TypeError('Wrong number of arguments for %s' % self.name)
        assert all(i is not None for i in args)

        brod = cupy.broadcast(*args)
        in_args = brod.values[:self.nin]
        out_args = list(args[self.nin:])
        if out is not None:
            assert len(out_args) == 0
            internal.check_args_device((out,))
            out_args = [out]
        internal.check_args_device(in_args + out_args)

        in_types, out_types, routine = self._guess_routine(in_args, dtype)

        out_args = _get_out_args(
            in_args, out_args, out_types, allocator, brod.shape)

        if len(out_args) == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in brod.shape:
            return ret

        # TODO(okuta): reorder dimension

        inout_args = in_args + out_args + [numpy.int32(brod.size)]
        for i, x in enumerate(inout_args):
            if isinstance(x, cupy.ndarray):
                inout_args[i] = x.reduced_view()
            elif i < len(in_args):
                inout_args[i] = in_types[i].type(x)

        params, kernel_args = _get_kernel_params_args(self._params, inout_args)
        is_ndarray = tuple(isinstance(x, cupy.ndarray) for x in in_args)
        raw_out_types = tuple(x.dtype for x in out_args)

        kern = _get_ufunc_kernel(
            in_types, out_types, tuple(raw_out_types),
            is_ndarray, params, routine, self.name, self._preamble)

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
                if all(t == dtype for t in out_types):
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
        in_types = tuple(numpy.dtype(t) for t in in_types)
        out_types = tuple(numpy.dtype(t) for t in out_types)
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

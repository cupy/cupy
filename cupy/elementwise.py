import string

import numpy
import six

import cupy
from cupy import carray
from cupy import cindexer
from cupy import cuda
from cupy import internal
from cupy import util


@util.memoize(for_each_device=True)
def _get_simple_elementwise_kernel(
        params, operation, name='kernel', options=(), preamble='',
        loop_prep='', after_loop=''):
    module_code = string.Template('''
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      CUPY_FOR(i, _ind.size()) {
        _ind.set(i);
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


_scaler_type = tuple(t.type for t in _typenames.keys())


def _get_typename(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    return _typenames[numpy.dtype(dtype)]


def _check_kernel_args(args):
    for a in args:
        assert isinstance(a, _scaler_type) or isinstance(
            a, (cupy.ndarray, cindexer.Indexer))


def _get_kernel_param_types(args):
    _check_kernel_args(args)
    ret = []
    for a in args:
        if isinstance(a, cindexer.Indexer):
            t = 'CIndexer<{}>'.format(a.ndim)
        else:
            t = _get_typename(a.dtype)
            if isinstance(a, cupy.ndarray):
                t = 'CArray<{}, {}>'.format(t, a.ndim)
        ret.append(t)
    return tuple(ret)


def _get_kernel_params(params, is_ndarray, param_types):
    return ', '.join(
        '{}{} {}{}'.format(
            'const ' if p.const else '',
            t,
            '_raw_' if f and not p.raw else '',
            p.name)
        for p, f, t in zip(params, is_ndarray, param_types))


def _reduce_dims(args, params, indexer):
    red = [a.reduced_view()
           if isinstance(a, cupy.ndarray) and not p.raw else None
           for a, p in zip(args, params)]
    max_arr = max(red, key=lambda x: 0 if x is None else x.ndim)
    if max_arr is None:
        return args, indexer
    try:
        for i, x in enumerate(red):
            if x is not None:
                x.shape = max_arr.shape
    except AttributeError:
        pass
    else:
        assert max_arr.size == indexer.size
        args = [a if r is None else r for a, r in zip(args, red)]
        indexer.shape = max_arr.shape
    return args, indexer


def _get_inout_args(args, indexer, params, reduce_dims):
    if reduce_dims:
        args, indexer = _reduce_dims(args, params, indexer)
    args += [indexer]
    is_ndarray = tuple(isinstance(x, cupy.ndarray) for x in args)
    return args, is_ndarray


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
        if t == 'CIndexer':
            pass
        elif len(t) == 1:
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


@util.memoize()
def _get_param_info(s, const=False):
    if len(s) == 0:
        return ()
    return tuple(ParameterInfo(i, const) for i in s.strip().split(','))


@util.memoize(for_each_device=True)
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


@util.memoize(for_each_device=True)
def _get_elementwise_kernel(
        params, is_ndarray, param_types, types, operation, name,
        options=(), preamble='', **kwargs):
    kernel_params = _get_kernel_params(params, is_ndarray, param_types)
    types_preamble = '\n'.join(
        'typedef {} {};'.format(_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for p, f in zip(params, is_ndarray):
        if not f or p.raw:
            continue
        if p.const:
            fmt = 'const {t} {n} = _raw_{n}[_ind.get()];'
        else:
            fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
        op.append(fmt.format(t=p.ctype, n=p.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        options=options, preamble=preamble, **kwargs)


class ElementwiseKernel(object):

    """User-defined elementwise kernel.

    This class can be used to define an elementwise kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ElementwiseKernel.__call__` method, which is cached for each device.
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
        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        param_rest = _get_param_info('CIndexer _ind')
        self.params = self.in_params + self.out_params + param_rest
        self.operation = operation
        self.name = name
        self.options = options
        self.reduce_dims = reduce_dims
        self.kwargs = kwargs
        names = [p.name for p in self.in_params + self.out_params]
        if 'i' in names:
            raise ValueError("Can not use 'i' as a parameter name")

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

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

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

        in_args = [x if isinstance(x, cupy.ndarray) else t.type(x)
                   for x, t in zip(in_args, in_types)]
        out_args = _get_out_args(
            in_args, out_args, out_types, allocator, brod.shape,
            self.out_params)

        if len(out_args) == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if n is None:
            indexer = cindexer.Indexer(brod.shape)
        else:
            indexer = cindexer.Indexer((n,))

        if brod.size == 0:
            return ret

        inout_args, is_ndarray = _get_inout_args(
            in_args + out_args, indexer, self.params, self.reduce_dims)
        param_types = _get_kernel_param_types(inout_args)
        kern = _get_elementwise_kernel(
            self.params, is_ndarray, param_types, types, self.operation,
            self.name, self.options, **self.kwargs)
        kern.linear_launch(indexer.size, inout_args)
        return ret


@util.memoize(for_each_device=True)
def _get_ufunc_kernel(in_types, out_types, out_raw_types, is_ndarray,
                      param_types, params, routine, name, preamble):
    kernel_params = _get_kernel_params(params, is_ndarray, param_types)

    types = []
    op = []
    for i, x in enumerate(in_types):
        types.append('typedef {} in{}_type;'.format(_get_typename(x), i))
        if not is_ndarray[i]:
            continue
        op.append(
            'const in{0}_type in{0} = _raw_in{0}[_ind.get()];'.format(i))

    for i, x in enumerate(out_types):
        types.append('typedef {} out{}_type;'.format(_get_typename(x), i))
        op.append('{1} &out{0} = _raw_out{0}[_ind.get()];'.format(
            i, _get_typename(out_raw_types[i])))

    op.append(routine)
    operation = '\n'.join(op)

    types.append(preamble)
    preamble = '\n'.join(types)

    return _get_simple_elementwise_kernel(
        kernel_params, operation, name, preamble=preamble)


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
        self.__doc__ = doc
        _in_params = tuple(
            ParameterInfo('T in{}'.format(i), True)
            for i in six.moves.range(nin))
        _out_params = tuple(
            ParameterInfo('T out{}'.format(i), False)
            for i in six.moves.range(nout))
        self._params = _in_params + _out_params + (
            ParameterInfo('CIndexer _ind', False),)

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

        in_args = [x if isinstance(x, cupy.ndarray) else t.type(x)
                   for x, t in zip(in_args, in_types)]
        out_args = _get_out_args(
            in_args, out_args, out_types, allocator, brod.shape)

        if len(out_args) == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in brod.shape:
            return ret

        indexer = cindexer.Indexer(brod.shape)
        inout_args, is_ndarray = _get_inout_args(
            in_args + out_args, indexer, self._params, True)
        param_types = _get_kernel_param_types(inout_args)
        out_raw_types = tuple(x.dtype for x in out_args)
        kern = _get_ufunc_kernel(
            in_types, out_types, out_raw_types,
            is_ndarray, param_types, self._params,
            routine, self.name, self._preamble)

        kern.linear_launch(indexer.size, inout_args)
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

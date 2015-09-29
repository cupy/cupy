import string

import numpy
import six

import cupy
from cupy import carray
from cupy import cuda
from cupy import util


six_range = six.moves.range
six_zip = six.moves.zip


def _get_simple_elementwise_kernel(
        params, operation, name, preamble,
        loop_prep='', after_loop='', options=()):
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

_python_scalar_type = six.integer_types + (float, bool)
_scalar_type = _python_scalar_type + tuple(
    t.type for t in _typenames.keys())

_kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
}


def _get_typename(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    return _typenames[numpy.dtype(dtype)]


def _check_args(args):
    dev = cuda.Device()
    cp_array = cupy.ndarray
    scalar_type = _scalar_type
    for arg in args:
        if isinstance(arg, cp_array):
            if arg.data.device != dev:
                raise ValueError('Array device must be same as the current '
                                 'device: array device = %d while current = %d'
                                 % (arg.device.id, dev.id))
        elif not isinstance(arg, scalar_type):
            raise TypeError('Unsupported type %s' % type(arg))


def _get_args_info(args):
    ret = []
    carray_Indexer = carray.Indexer
    ret_append = ret.append
    for a in args:
        t = type(a)
        if t == carray_Indexer:
            dtype = None
        else:
            dtype = a.dtype.type
        ret_append((t, dtype, a.ndim))
    return tuple(ret)


def _get_kernel_params(params, args_info):
    ret = []
    for p, a in six_zip(params, args_info):
        type, dtype, ndim = a
        is_array = type is cupy.ndarray
        if type is carray.Indexer:
            t = 'CIndexer<%d>' % ndim
        else:
            t = _get_typename(dtype)
            if is_array:
                t = 'CArray<%s, %d>' % (t, ndim)
        ret.append('%s%s %s%s' % ('const ' if p.is_const else '',
                                  t,
                                  '_raw_' if is_array and not p.raw else '',
                                  p.name))
    return ', '.join(ret)


def _reduce_dims(args, params, shape):
    ndim = len(shape)
    if ndim <= 1:
        return args, shape

    cp_array = cupy.ndarray
    is_array_flags = [not p.raw and isinstance(a, cp_array)
                      for p, a in six_zip(params, args)]
    args_strides = [a._strides for a, f in six_zip(args, is_array_flags) if f]

    src_shape = shape
    shape = list(src_shape)
    cnt = 0
    for i in six_range(1, ndim):
        j = i - 1
        shape_i = shape[i]
        shape_j = shape[j]
        if shape_j == 1:
            continue
        for strides in args_strides:
            if strides[i] * shape_i != strides[j]:
                cnt += 1
                axis = j
                break
        else:
            shape[i] *= shape_j
            shape[j] = 1
    if shape[-1] != 1:
        cnt += 1
        axis = -1

    if not cnt:
        return args, src_shape
    elif cnt == 1:
        new_shape = shape[axis],
        args = list(args)
        for i, a in enumerate(args):
            if is_array_flags[i]:
                a = args[i] = a.view()
                a._shape = new_shape
                a._strides = a._strides[axis],
        return args, new_shape

    new_shape = tuple([dim for dim in shape if dim != 1])
    args = list(args)
    for i, a in enumerate(args):
        if is_array_flags[i]:
            a = args[i] = a.view()
            a._shape = new_shape
            a._strides = tuple(
                [st for st, sh in six_zip(a._strides, shape) if sh != 1])
    return args, new_shape


class ParameterInfo(object):

    def __init__(self, str, is_const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.is_const = is_const
        s = tuple(i for i in str.split() if len(i) != 0)
        if len(s) < 2:
            raise Exception('Syntax error: %s' % str)

        t, self.name = s[-2:]
        if t == 'CIndexer':
            pass
        elif len(t) == 1:
            self.ctype = t
        else:
            dtype = numpy.dtype(t)
            self.dtype = dtype.type
            if dtype.name != t:
                raise ValueError('Wrong type %s' % t)
            self.ctype = _get_typename(self.dtype)

        for i in s[:-2]:
            if i == 'raw':
                self.raw = True
            else:
                raise Exception('Unknown keyward "%s"' % i)


@util.memoize()
def _get_param_info(s, is_const):
    if len(s) == 0:
        return ()
    return tuple([ParameterInfo(i, is_const) for i in s.strip().split(',')])


@util.memoize()
def _decide_params_type(in_params, out_params, in_args_dtype, out_args_dtype):
    type_dict = {}
    if out_args_dtype:
        assert len(out_params) == len(out_args_dtype)
        for p, a in six_zip(out_params, out_args_dtype):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if a != p.dtype:
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if t != a:
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_args_dtype)
    unknown_ctype = []
    for p, a in six_zip(in_params, in_args_dtype):
        if a is None:
            if p.dtype is None:
                unknown_ctype.append(p.ctype)
        else:
            if p.dtype is not None:
                if a != p.dtype:
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if t != a:
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    in_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                      for p in in_params])
    out_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                       for p in out_params])
    return in_types, out_types, tuple(type_dict.items())


def _broadcast(args, params, use_size):
    value = [a if not p.raw and isinstance(a, cupy.ndarray) else None
             for p, a in six_zip(params, args)]
    if use_size:
        for i in value:
            if i is None:
                break
        else:
            raise ValueError("Specified 'size' can be used only "
                             "if all of the ndarray are 'raw'.")
    else:
        for i in value:
            if i is not None:
                break
        else:
            raise ValueError('Loop size is Undecided')
    brod = cupy.broadcast(*value)
    value = [b if a is None else a
             for a, b in six_zip(brod.values, args)]
    return value, brod.shape


def _get_out_args(out_args, out_types, out_shape):
    if not out_args:
        return [cupy.empty(out_shape, t) for t in out_types]

    for a in out_args:
        if not isinstance(a, cupy.ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if a.shape != out_shape:
            raise ValueError('Out shape is mismatched')
    return out_args


def _get_out_args_with_params(out_args, out_types, out_shape, out_params):
    if not out_args:
        for p in out_params:
            if p.raw:
                raise ValueError('Output array size is Undecided')
        return [cupy.empty(out_shape, t) for t in out_types]

    for a, p in six_zip(out_args, out_params):
        if not isinstance(a, cupy.ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if a.shape != out_shape and not p.raw:
            raise ValueError('Out shape is mismatched')
    return out_args


@util.memoize(for_each_device=True)
def _get_elementwise_kernel(args_info, types, params, operation, name,
                            preamble, kwargs):
    kernel_params = _get_kernel_params(params, args_info)
    types_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for p, a in six_zip(params, args_info):
        if not p.raw and a[0] == cupy.ndarray:
            if p.is_const:
                fmt = 'const {t} {n} = _raw_{n}[_ind.get()];'
            else:
                fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
            op.append(fmt.format(t=p.ctype, n=p.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        preamble, **dict(kwargs))


class ElementwiseKernel(object):

    """User-defined elementwise kernel.

    This class can be used to define an elementwise kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ElementwiseKernel.__call__` method,
    which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        operation (str): The body in the loop written in CUDA-C/C++.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_dims (bool): If False, the shapes of array arguments are
            kept within the kernel invocation. The shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            ndims) by default. It may make the kernel fast by reducing the
            index calculations.
        options (list): Options passed to the nvcc command.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """
    def __init__(self, in_params, out_params, operation,
                 name='kernel', reduce_dims=True, preamble='', **kwargs):
        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        param_rest = _get_param_info('CIndexer _ind', False)
        self.params = self.in_params + self.out_params + param_rest
        self.operation = operation
        self.name = name
        self.reduce_dims = reduce_dims
        self.preamble = preamble
        self.kwargs = frozenset(kwargs.items())
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
        size = kwargs.pop('size', None)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)
        _check_args(args)

        values, shape = _broadcast(args, self.params, size is not None)
        in_args = values[:self.nin]
        out_args = values[self.nin:]

        cp_array = cupy.ndarray
        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, cp_array) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, cp_array) else None
             for a in out_args])

        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)

        out_args = _get_out_args_with_params(
            out_args, out_types, shape, self.out_params)
        if self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if size is not None:
            shape = size,

        if 0 in shape:
            return ret

        inout_args = [x if isinstance(x, cp_array) else t(x)
                      for x, t in six_zip(in_args, in_types)]
        inout_args += out_args

        if self.reduce_dims:
            inout_args, shape = _reduce_dims(
                inout_args, self.params, shape)
        indexer = carray.Indexer(shape)
        inout_args.append(indexer)

        args_info = _get_args_info(inout_args)
        kern = _get_elementwise_kernel(
            args_info, types, self.params, self.operation,
            self.name, self.preamble, self.kwargs)
        kern.linear_launch(indexer.size, inout_args)
        return ret


@util.memoize(for_each_device=True)
def _get_ufunc_kernel(in_types, out_types, routine, args_info, out_raw_types,
                      params, name, preamble):
    kernel_params = _get_kernel_params(params, args_info)

    types = []
    op = []
    for i, x in enumerate(in_types):
        types.append('typedef %s in%d_type;' % (_get_typename(x), i))
        if args_info[i][0] is cupy.ndarray:
            op.append(
                'const in{0}_type in{0} = _raw_in{0}[_ind.get()];'.format(i))

    for i, x in enumerate(out_types):
        types.append('typedef %s out%d_type;' % (_get_typename(x), i))
        op.append('{1} &out{0} = _raw_out{0}[_ind.get()];'.format(
            i, _get_typename(out_raw_types[i])))

    op.append(routine)
    operation = '\n'.join(op)

    types.append(preamble)
    preamble = '\n'.join(types)

    return _get_simple_elementwise_kernel(
        kernel_params, operation, name, preamble)


def _guess_routine_from_in_types(ops, in_types):
    for op in ops:
        for dst, src in six_zip(op[0], in_types):
            if not numpy.can_cast(src, dst):
                break
        else:
            return op
    return None


def _guess_routine_from_dtype(ops, dtype):
    for op in ops:
        for t in op[1]:
            if t != dtype:
                break
        else:
            return op
    return None


def _check_in_args_kind(in_args):
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for i in in_args:
        if isinstance(i, cupy.ndarray):
            kind = _kind_score[i.dtype.kind]
            all_scalars = False
            if kind > max_array_kind:
                max_array_kind = kind
        else:
            if isinstance(i, _python_scalar_type):
                dtype = numpy.dtype(type(i))
            else:
                dtype = i.dtype
            kind = _kind_score[dtype.kind]
            if kind > max_scalar_kind:
                max_scalar_kind = kind
    return not all_scalars and max_array_kind >= max_scalar_kind


def _guess_routine(name, cache, ops, in_args, dtype):
    if dtype is None:
        use_raw_value = _check_in_args_kind(in_args)
        if use_raw_value:
            in_types = tuple(in_args)
            op = ()
        else:
            in_types = tuple(
                [type(i)
                 if isinstance(i, _python_scalar_type) else i.dtype.type
                 for i in in_args])
            op = cache.get(in_types, ())

        if op is ():
            op = _guess_routine_from_in_types(ops, in_types)
            if not use_raw_value:
                cache[in_types] = op
    else:
        op = cache.get(dtype, ())
        if op is ():
            op = _guess_routine_from_dtype(ops, dtype)
            cache[dtype] = op

    if op:
        return op
    raise TypeError('Wrong type of arguments for %s' % name)


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
            ParameterInfo('T in%d' % i, True)
            for i in six_range(nin))
        _out_params = tuple(
            ParameterInfo('T out%d' % i, False)
            for i in six_range(nout))
        self._params = _in_params + _out_params + (
            ParameterInfo('CIndexer _ind', False),)
        self._routine_cache = {}

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
            in_str = ''.join([numpy.dtype(t).char for t in in_types])
            out_str = ''.join([numpy.dtype(t).char for t in out_types])
            types.append('%s->%s' % (in_str, out_str))
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

        Returns:
            Output array or a tuple of output arrays.

        """
        out = kwargs.pop('out', None)
        dtype = kwargs.pop('dtype', None)
        if dtype is not None:
            dtype = numpy.dtype(dtype).type
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if out is None:
            in_args = args[:self.nin]
            out_args = args[self.nin:]
        else:
            if self.nout != 1:
                raise ValueError("Cannot use 'out' in %s" % self.name)
            if n_args != self.nin:
                raise ValueError("Cannot specify 'out' as both "
                                 "a positional and keyword argument")
            in_args = args
            out_args = out,
            args += out_args

        _check_args(args)
        broad = cupy.broadcast(*args)
        shape = broad.shape

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        out_args = _get_out_args(out_args, out_types, shape)
        if self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in shape:
            return ret

        inout_args = [x if isinstance(x, cupy.ndarray) else t(x)
                      for x, t in six_zip(broad.values, in_types)]
        inout_args.extend(out_args)
        inout_args, shape = _reduce_dims(inout_args, self._params, shape)
        indexer = carray.Indexer(shape)
        inout_args.append(indexer)
        args_info = _get_args_info(inout_args)
        out_raw_types = tuple([x.dtype.type for x in out_args])

        kern = _get_ufunc_kernel(
            in_types, out_types, routine,
            args_info, out_raw_types,
            self._params, self.name, self._preamble)

        kern.linear_launch(indexer.size, inout_args)
        return ret


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
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble, doc)


_id = 'out0 = in0'

copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    _id)


copy_where = create_ufunc(
    'cupy_copy_where',
    ('??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d'),
    'if (in1) out0 = in0')


_divmod = create_ufunc(
    'cupy_divmod',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0_type a = _floor_divide(in0, in1); out0 = a; out1 = in0 - a * in1')

import string
import warnings

import numpy

import cupy
from cupy.cuda import compiler
from cupy import _util

cimport cython  # NOQA

from libcpp cimport vector

from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda cimport memory
from cupy.cuda cimport texture
from cupy._core cimport _accelerator
from cupy._core cimport _carray
from cupy._core cimport _scalar
from cupy._core._dtype cimport get_dtype, _raise_if_invalid_cast
from cupy._core._memory_range cimport may_share_bounds
from cupy._core._scalar import get_typename as _get_typename
from cupy._core cimport core
from cupy._core.core cimport _convert_object_with_cuda_array_interface
from cupy._core.core cimport _ndarray_init
from cupy._core.core cimport compile_with_cache
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal
from cupy_backends.cuda.api cimport runtime

try:
    import cupy_backends.cuda.libs.cutensor as cuda_cutensor
except ImportError:
    cuda_cutensor = None

from cupy._core import _fusion_thread_local


cdef inline bint _contains_zero(const shape_t& v) except? -1:
    for i in range(v.size()):
        if v[i] == 0:
            return True
    return False


@_util.memoize(for_each_device=True)
def _get_warpsize():
    device_id = runtime.getDevice()
    return runtime.getDeviceProperties(device_id)['warpSize']


cdef str _get_simple_elementwise_kernel_code(
        tuple params, tuple arginfos, str operation, str name,
        _TypeMap type_map, str preamble, str loop_prep='', str after_loop=''):
    # No loop unrolling due to avoid 64-bit division
    module_code = string.Template('''
    ${typedef_preamble}
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      #pragma unroll 1
      CUPY_FOR(i, _ind.size()) {
        _ind.set(i);
        ${operation};
      }
      ${after_loop};
    }
    ''').substitute(
        typedef_preamble=type_map.get_typedef_code(),
        params=_get_kernel_params(params, arginfos),
        operation=operation,
        name=name,
        preamble=preamble,
        loop_prep=loop_prep,
        after_loop=after_loop)
    return module_code


cdef function.Function _get_simple_elementwise_kernel_from_code(
        str name, str code, tuple options=()):
    module = compile_with_cache(code, options)
    return module.get_function(name)


cdef function.Function _get_simple_elementwise_kernel(
        tuple params, tuple arginfos, str operation, str name,
        _TypeMap type_map, str preamble, str loop_prep='', str after_loop='',
        tuple options=()):
    code = _get_simple_elementwise_kernel_code(
        params, arginfos, operation, name, type_map, preamble, loop_prep,
        after_loop
    )
    return _get_simple_elementwise_kernel_from_code(name, code, options)


cdef inline int _get_kind_score(int kind):
    if b'b' == kind:
        return 0
    if b'u' == kind or b'i' == kind:
        return 1
    if b'f' == kind or b'c' == kind:
        return 2
    return -1


@cython.profile(False)
cpdef inline _check_peer_access(_ndarray_base arr, int device_id):
    if arr.data.device_id == device_id:
        return

    msg = (
        f'The device where the array resides ({arr.data.device_id}) is '
        f'different from the current device ({device_id}).'
    )

    cdef bint peer_access = device._enable_peer_access(
        device_id, arr.data.device_id)
    if not peer_access:
        raise ValueError(
            f'{msg} Peer access is unavailable between these devices.')
    warnings.warn(
        f'{msg} Peer access has been activated automatically.',
        _util.PerformanceWarning)


cdef inline _preprocess_arg(int dev_id, arg, bint use_c_scalar):
    if isinstance(arg, _ndarray_base):
        s = arg
        _check_peer_access(<_ndarray_base>s, dev_id)
    elif isinstance(arg, texture.TextureObject):
        s = arg
    elif hasattr(arg, '__cuda_array_interface__'):
        s = _convert_object_with_cuda_array_interface(arg)
        _check_peer_access(<_ndarray_base>s, dev_id)
    elif hasattr(arg, '__cupy_get_ndarray__'):
        s = arg.__cupy_get_ndarray__()
        _check_peer_access(<_ndarray_base>s, dev_id)
    else:  # scalars or invalid args
        if use_c_scalar:
            s = _scalar.scalar_to_c_scalar(arg)
        else:
            s = _scalar.scalar_to_numpy_scalar(arg)
        if s is None:
            raise TypeError('Unsupported type %s' % type(arg))
    return s


cdef list _preprocess_args(int dev_id, args, bint use_c_scalar):
    """Preprocesses arguments for kernel invocation

    - Checks device compatibility for ndarrays
    - Converts Python/NumPy scalars:
      - If use_c_scalar is True, into CScalars.
      - If use_c_scalar is False, into NumPy scalars.
    """
    cdef list ret = []
    for arg in args:
        ret.append(_preprocess_arg(dev_id, arg, use_c_scalar))
    return ret


cdef list _preprocess_optional_args(int dev_id, args, bint use_c_scalar):
    """Preprocesses arguments for kernel invocation

    - Checks device compatibility for ndarrays
    - Converts Python/NumPy scalars:
      - If use_c_scalar is True, into CScalars.
      - If use_c_scalar is False, into NumPy scalars.
    """
    cdef list ret = []
    for arg in args:
        if arg is None:
            ret.append(None)
        else:
            ret.append(_preprocess_arg(dev_id, arg, use_c_scalar))
    return ret


cdef class _ArgInfo:
    # Holds metadata of an argument.
    # This class is immutable and used as a part of hash keys.

    def __init__(self, *args):
        arg_kind, typ, dtype, ndim, c_contiguous, index_32_bits = args
        self._init(arg_kind, typ, dtype, ndim, c_contiguous, index_32_bits)

    cdef _ArgInfo _init(
            self,
            _ArgKind arg_kind,
            type typ,
            object dtype,
            int ndim,
            bint c_contiguous,
            bint index_32_bits):
        self.arg_kind = arg_kind
        self.type = typ
        self.dtype = dtype
        self.ndim = ndim
        self.c_contiguous = c_contiguous
        self.index_32_bits = index_32_bits

    @staticmethod
    cdef _ArgInfo from_arg(object arg):
        typ = type(arg)
        if issubclass(typ, _ndarray_base):
            return _ArgInfo.from_ndarray(arg)
        if typ is _scalar.CScalar:
            return _ArgInfo.from_scalar(arg)
        if typ is _carray.Indexer:
            return _ArgInfo.from_indexer(arg)
        if typ is memory.MemoryPointer:
            return _ArgInfo.from_memptr(arg)
        if typ is texture.TextureObject:
            return _ArgInfo.from_texture(arg)
        assert False, typ

    @staticmethod
    cdef _ArgInfo from_ndarray(_ndarray_base arg):
        cdef _ArgInfo ret = _ArgInfo.__new__(_ArgInfo)
        ret._init(
            ARG_KIND_NDARRAY,
            type(arg),
            arg.dtype.type,
            arg._shape.size(),
            arg._c_contiguous,
            arg._index_32_bits)
        return ret

    @staticmethod
    cdef _ArgInfo from_scalar(_scalar.CScalar arg):
        cdef _ArgInfo ret = _ArgInfo.__new__(_ArgInfo)
        dtype = arg.get_numpy_type()
        ret._init(ARG_KIND_SCALAR, _scalar.CScalar, dtype, 0, True, True)
        return ret

    @staticmethod
    cdef _ArgInfo from_indexer(_carray.Indexer arg):
        cdef _ArgInfo ret = _ArgInfo.__new__(_ArgInfo)
        ret._init(
            ARG_KIND_INDEXER, _carray.Indexer, None, arg.ndim, True,
            arg._index_32_bits)
        return ret

    @staticmethod
    cdef _ArgInfo from_memptr(memory.MemoryPointer arg):
        cdef _ArgInfo ret = _ArgInfo.__new__(_ArgInfo)
        ret._init(
            ARG_KIND_POINTER, memory.MemoryPointer, None, 0, True, True)
        return ret

    @staticmethod
    cdef _ArgInfo from_texture(texture.TextureObject arg):
        cdef _ArgInfo ret = _ArgInfo.__new__(_ArgInfo)
        ret._init(
            ARG_KIND_TEXTURE, texture.TextureObject, None, 0, True, True)
        return ret

    def __hash__(self):
        return hash((self.arg_kind, self.type, self.dtype, self.ndim,
                     self.c_contiguous, self.index_32_bits))

    def __eq__(self, other):
        cdef _ArgInfo oth
        if not isinstance(other, _ArgInfo):
            return False
        oth = other
        return (
            self.arg_kind == oth.arg_kind
            and self.type is oth.type
            and self.dtype == oth.dtype
            and self.ndim == oth.ndim
            and self.c_contiguous == oth.c_contiguous
            and self.index_32_bits == oth.index_32_bits)

    def __repr__(self):
        return '<_ArgInfo({})>'.format(
            ' '.join([
                'arg_kind={!r}'.format(self.arg_kind),
                'type={!r}'.format(self.type),
                'dtype={!r}'.format(self.dtype),
                'ndim={!r}'.format(self.ndim),
                'c_contiguous={!r}'.format(self.c_contiguous),
                'index_32_bits={!r}'.format(self.index_32_bits),
            ]))

    cdef _ArgInfo as_ndarray_with_ndim(self, int ndim):
        # Returns an ndarray _ArgInfo with altered ndim.
        # If ndim is the same, self is returned untouched.
        assert self.arg_kind == ARG_KIND_NDARRAY
        if self.ndim == ndim:
            return self
        return _ArgInfo(
            ARG_KIND_NDARRAY, self.dtype, self.dtype, ndim, False, False)

    cdef bint is_ndarray(self):
        return self.arg_kind == ARG_KIND_NDARRAY

    cdef bint is_scalar(self):
        return self.arg_kind == ARG_KIND_SCALAR

    cdef str get_c_type(self):
        # Returns the C type representation.
        if self.arg_kind == ARG_KIND_NDARRAY:
            return 'CArray<%s, %d, %d, %d>' % (
                _get_typename(self.dtype), self.ndim,
                self.c_contiguous, self.index_32_bits)
        if self.arg_kind == ARG_KIND_SCALAR:
            return _get_typename(self.dtype)
        if self.arg_kind == ARG_KIND_INDEXER:
            return 'CIndexer<%d, %d>' % (self.ndim, self.index_32_bits)
        if self.arg_kind == ARG_KIND_TEXTURE:
            return 'cudaTextureObject_t'
        assert False

    cdef str get_param_c_type(self, ParameterInfo p):
        # Returns the C type representation in the global function's
        # parameter list.
        cdef str ctyp = self.get_c_type()
        if p.is_const:
            return 'const ' + ctyp
        return ctyp

    cdef str get_c_var_name(self, ParameterInfo p):
        if self.arg_kind in (ARG_KIND_NDARRAY, ARG_KIND_POINTER) and not p.raw:
            return '_raw_' + p.name
        return p.name


cdef tuple _get_arginfos(list args):
    return tuple([_ArgInfo.from_arg(a) for a in args])


cdef str _get_kernel_params(tuple params, tuple arginfos):
    cdef ParameterInfo p
    cdef _ArgInfo arginfo
    assert len(params) == len(arginfos)
    lst = []
    for i in range(len(params)):
        p = params[i]
        arginfo = arginfos[i]
        lst.append('{} {}'.format(
            arginfo.get_param_c_type(p),
            arginfo.get_c_var_name(p)))
    return ', '.join(lst)


cdef shape_t _reduce_dims(list args, tuple params, const shape_t& shape):
    """ Remove contiguous stride to optimize CUDA kernel."""
    cdef _ndarray_base arr

    if shape.size() <= 1 or len(args) == 0:
        return shape

    if len(args) == 1:  # fast path for reduction
        a = args[0]
        if (<ParameterInfo>params[0]).raw or not isinstance(a, _ndarray_base):
            return shape
        arr = a
        arr = arr.reduced_view()
        if arr is a:
            return shape
        else:
            args[0] = arr
            return arr._shape
    return _reduced_view_core(args, params, shape)


cdef shape_t _reduced_view_core(list args, tuple params, const shape_t& shape):
    cdef int i, ax, last_ax, ndim
    cdef Py_ssize_t total_size
    cdef shape_t vecshape, newshape, newstrides
    cdef vector.vector[int] array_indexes, axes
    cdef vector.vector[int] strides_indexes
    cdef ParameterInfo p
    cdef _ndarray_base arr

    ndim = shape.size()
    array_indexes.reserve(len(args))
    strides_indexes.reserve(len(args))
    for i in range(len(args)):
        p = params[i]
        if p.raw:
            continue
        a = args[i]
        if isinstance(a, _ndarray_base):
            array_indexes.push_back(i)
            arr = a
            if not arr._c_contiguous:
                if ndim == 2:  # short cut
                    return shape
                strides_indexes.push_back(i)

    if array_indexes.size() == 0:
        return shape

    if strides_indexes.size() == 0:
        # The input arrays are all c_contiguous
        i = array_indexes[0]
        arr = args[i]
        total_size = arr.size
        newshape.assign(<Py_ssize_t>1, total_size)
        newstrides.resize(1)
        for i in array_indexes:
            arr = args[i]
            newstrides[0] = arr.dtype.itemsize
            # TODO(niboshi): Confirm update_x_contiguity flags
            args[i] = arr._view(
                type(arr), newshape, newstrides, False, True, arr)
        return newshape

    axes.reserve(ndim)
    vecshape.reserve(ndim)
    for ax in range(ndim):
        vecshape.push_back(shape[ax])
    last_ax = -1
    for ax in range(ndim):
        if vecshape[ax] == 1:
            continue
        if last_ax < 0:
            last_ax = ax
            continue
        for i in strides_indexes:
            arr = args[i]
            if arr._strides[ax] * vecshape[ax] != arr._strides[last_ax]:
                axes.push_back(last_ax)
                break
        else:
            vecshape[ax] *= vecshape[last_ax]
        last_ax = ax
    if last_ax >= 0:
        axes.push_back(last_ax)
    if <int>axes.size() == ndim:
        return shape

    newshape.reserve(axes.size())
    newstrides.reserve(axes.size())
    for ax in axes:
        newshape.push_back(vecshape[ax])
    for i in array_indexes:
        arr = args[i]
        newstrides.clear()
        for ax in axes:
            newstrides.push_back(arr._strides[ax])
        # TODO(niboshi): Confirm update_x_contiguity flags
        args[i] = arr._view(type(arr), newshape, newstrides, False, True, arr)
    return newshape


cdef class ParameterInfo:

    def __init__(self, str param, bint is_const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.is_const = is_const
        s = tuple([i for i in param.split() if len(i) != 0])
        if len(s) < 2:
            raise Exception('Syntax error: %s' % param)

        t, self.name = s[-2:]
        if t == 'CIndexer':
            pass
        elif len(t) == 1:
            self.ctype = t
        else:
            dtype = get_dtype(t)
            self.dtype = dtype.type
            if dtype.name != t:
                raise ValueError('Wrong type %s' % t)
            self.ctype = _get_typename(self.dtype)

        for i in s[:-2]:
            if i == 'raw':
                self.raw = True
            elif i == '_non_const':
                self.is_const = False
            else:
                raise Exception('Unknown keyword "%s"' % i)

    def __hash__(self):
        return hash((
            self.name, self.dtype, self.ctype, self.raw, self.is_const))

    def __eq__(self, other):
        cdef ParameterInfo oth
        if not isinstance(other, ParameterInfo):
            return False
        oth = other
        return (
            self.name == oth.name
            and self.dtype == oth.dtype
            and self.ctype == oth.ctype
            and self.raw == oth.raw
            and self.is_const == oth.is_const)

    def __repr__(self):
        return '<ParameterInfo({})>'.format(
            ' '.join([
                'name={!r}'.format(self.name),
                'dtype={!r}'.format(self.dtype),
                'ctype={!r}'.format(self.ctype),
                'raw={!r}'.format(self.raw),
                'is_const={!r}'.format(self.is_const),
            ]))


@_util.memoize()
def _get_param_info(str s, is_const):
    if len(s) == 0:
        return ()
    return tuple([ParameterInfo(i, is_const) for i in s.strip().split(',')])


@_util.memoize()
def _decide_params_type(in_params, out_params, in_args_dtype, out_args_dtype):
    return _decide_params_type_core(in_params, out_params, in_args_dtype,
                                    out_args_dtype)


cdef class _TypeMap:

    def __init__(self, pairs):
        self._pairs = pairs

    def __hash__(self):
        return hash(self._pairs)

    def __eq__(self, other):
        if not isinstance(other, _TypeMap):
            return False
        return self._pairs == (<_TypeMap>other)._pairs

    def __str__(self):
        return '<_TypeMap {}>'.format(self._pairs)

    cdef str get_typedef_code(self):
        # Returns a code fragment of typedef statements used as preamble.
        return ''.join([
            'typedef %s %s;\n' % (_get_typename(ctype2), ctype1)
            for ctype1, ctype2 in self._pairs])


cdef tuple _decide_params_type_core(
        tuple in_params, tuple out_params, tuple in_args_dtype,
        tuple out_args_dtype):
    type_dict = {}
    if out_args_dtype:
        assert len(out_params) == len(out_args_dtype)
        for p, a in zip(out_params, out_args_dtype):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if get_dtype(a) != get_dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if get_dtype(t) != get_dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_args_dtype)
    unknown_ctype = []  # TODO(leofang): remove this as it's unused?
    for p, a in zip(in_params, in_args_dtype):
        if a is None:
            if p.dtype is None:
                unknown_ctype.append(p.ctype)
        else:
            if p.dtype is not None:
                if numpy.dtype(a) != numpy.dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if numpy.dtype(t) != numpy.dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    in_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                      for p in in_params])
    out_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                       for p in out_params])
    type_map = _TypeMap(tuple(sorted(type_dict.items())))
    return in_types, out_types, type_map


cdef list _broadcast(list args, tuple params, bint use_size, shape_t& shape):
    # `shape` is an output argument
    cdef Py_ssize_t i
    cdef ParameterInfo p
    cdef bint any_nonraw_array = False

    # Collect non-raw arrays
    value = []
    for i, a in enumerate(args):
        p = params[i]
        if not p.raw and isinstance(a, _ndarray_base):
            # Non-raw array
            any_nonraw_array = True
            value.append(a)
        else:
            value.append(None)

    if use_size:
        if any_nonraw_array:
            raise ValueError('Specified \'size\' can be used only '
                             'if all of the ndarray are \'raw\'.')
    else:
        if not any_nonraw_array:
            raise ValueError('Loop size is undecided.')

    # Perform broadcast.
    # Note that arrays in `value` are replaced with broadcasted ones.
    internal._broadcast_core(value, shape)

    # Restore raw arrays and scalars from the original list.
    for i, a in enumerate(value):
        if a is None:
            value[i] = args[i]
    return value


cdef _numpy_can_cast = numpy.can_cast


cdef list _get_out_args_from_optionals(
    subtype, list out_args, tuple out_types, const shape_t& out_shape, casting,
    obj
):
    cdef _ndarray_base arr

    while len(out_args) < len(out_types):
        out_args.append(None)

    for i, a in enumerate(out_args):
        if a is None:
            out_args[i] = _ndarray_init(
                subtype, out_shape, out_types[i], obj)
            continue

        if not isinstance(a, _ndarray_base):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        arr = a
        if not internal.vector_equal(arr._shape, out_shape):
            raise ValueError('Out shape is mismatched')
        out_type = get_dtype(out_types[i])

        _raise_if_invalid_cast(out_type, arr.dtype, casting, "output operand")
    return out_args


cdef _copy_in_args_if_needed(list in_args, list out_args):
    # `in_args` is an input and output argument
    cdef _ndarray_base inp, out
    for i in range(len(in_args)):
        a = in_args[i]
        if isinstance(a, _ndarray_base):
            inp = a
            for out in out_args:
                if inp is not out and may_share_bounds(inp, out):
                    in_args[i] = inp.copy()
                    break


cdef list _get_out_args_with_params(
        list out_args, tuple out_types, const shape_t& out_shape,
        tuple out_params, bint is_size_specified):
    cdef ParameterInfo p
    cdef _ndarray_base arr
    if not out_args:
        for p in out_params:
            if p.raw and not is_size_specified:
                raise ValueError('Output array size is Undecided')
        return [_ndarray_init(
            cupy.ndarray, out_shape, t, None) for t in out_types]

    for i, p in enumerate(out_params):
        a = out_args[i]
        if not isinstance(a, _ndarray_base):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        arr = a
        if not p.raw and not internal.vector_equal(arr._shape, out_shape):
            raise ValueError('Out shape is mismatched')
    return out_args


@_util.memoize()
def _get_elementwise_kernel_code(
        tuple arginfos, _TypeMap type_map,
        tuple params, str operation, str name,
        str preamble, str loop_prep='', str after_loop='', tuple options=()):
    cdef _ArgInfo arginfo

    op = []
    for p, arginfo in zip(params, arginfos):
        if arginfo.is_ndarray() and not p.raw:
            if p.is_const:
                fmt = 'const {t} &{n} = _raw_{n}[_ind.get()];'
            else:
                fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
            op.append(fmt.format(t=p.ctype, n=p.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel_code(
        params, arginfos, operation, name, type_map,
        preamble, loop_prep, after_loop)


@_util.memoize(for_each_device=True)
def _get_elementwise_kernel(
        tuple arginfos, _TypeMap type_map,
        tuple params, str operation, str name,
        str preamble, str loop_prep='', str after_loop='', tuple options=()):
    cdef str code = _get_elementwise_kernel_code(
        arginfos, type_map, params, operation, name, preamble, loop_prep,
        after_loop
    )
    return _get_simple_elementwise_kernel_from_code(name, code, options)


cdef class ElementwiseKernel:

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
        reduce_dims (bool): If ``False``, the shapes of array arguments are
            kept within the kernel invocation. The shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            dimension) by default. It may make the kernel fast by reducing the
            index calculations.
        options (tuple): Compile options passed to NVRTC. For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        no_return (bool): If ``True``, __call__ returns ``None``.
        return_tuple (bool): If ``True``, __call__ always returns tuple of
            array even if single value is returned.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """

    cdef:
        readonly tuple in_params
        readonly tuple out_params
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly tuple params
        readonly object operation
        readonly str name
        readonly str __name__
        readonly bint reduce_dims
        readonly object preamble
        readonly bint no_return
        readonly bint return_tuple
        readonly dict kwargs
        readonly dict _params_type_memo
        readonly dict _elementwise_kernel_memo
        readonly dict _cached_codes

    def __init__(self, in_params, out_params, operation,
                 name='kernel', reduce_dims=True, preamble='',
                 no_return=False, return_tuple=False, **kwargs):
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

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
        self.no_return = no_return
        self.return_tuple = return_tuple
        self.kwargs = kwargs
        self._params_type_memo = {}
        self._cached_codes = {}
        names = [p.name for p in self.in_params + self.out_params]
        if 'i' in names:
            raise ValueError('Can not use \'i\' as a parameter name')
        self._elementwise_kernel_memo = {}
        # This is for profiling mechanisms to auto infer a name
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or dimensions are not
        compatible. It means that single ElementwiseKernel object may be
        compiled into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            size (int): Range size of the indices.  By default, the range size
                is automatically determined from the result of broadcasting.
                This parameter must be specified if and only if all ndarrays
                are `raw` and the range size cannot be determined
                automatically.
            block_size (int): Number of threads per block. By default, the
                value is set to 128.

        Returns:
            If ``no_return`` has not set, arrays are returned according to the
            ``out_params`` argument of the ``__init__`` method.
            If ``no_return`` has set, ``None`` is returned.

        """
        cdef function.Function kern
        cdef Py_ssize_t size, i
        cdef list in_args, out_args
        cdef tuple in_types, out_types
        cdef shape_t shape

        size = kwargs.pop('size', -1)
        stream = kwargs.pop('stream', None)
        block_size = kwargs.pop('block_size', 128)
        if len(kwargs):
            raise TypeError('Wrong arguments %s' % kwargs)
        if block_size <= 0:
            raise ValueError('block_size must be greater than zero')
        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError(
                'Wrong number of arguments for {!r}. '
                'It must be either {} or {} (with outputs), '
                'but given {}.'.format(
                    self.name, self.nin, self.nargs, n_args))
        for arg in args:
            if hasattr(arg, '__cupy_override_elementwise_kernel__'):
                return arg.__cupy_override_elementwise_kernel__(
                    self, *args, **kwargs)
        dev_id = device.get_device_id()
        arg_list = _preprocess_args(dev_id, args, True)

        out_args = arg_list[self.nin:]
        # _broadcast updates shape
        in_args = _broadcast(
            arg_list, self.params, size != -1, shape)[:self.nin]

        in_ndarray_types = []
        for a in in_args:
            if isinstance(a, _ndarray_base):
                t = a.dtype.type
            elif isinstance(a, texture.TextureObject):
                t = 'cudaTextureObject_t'
            else:
                t = None
            in_ndarray_types.append(t)
        in_ndarray_types = tuple(in_ndarray_types)
        out_ndarray_types = tuple([a.dtype.type for a in out_args])

        in_types, out_types, type_map = self._decide_params_type(
            in_ndarray_types, out_ndarray_types)

        is_size_specified = False
        if size != -1:
            shape.assign(1, size)
            is_size_specified = True

        out_args = _get_out_args_with_params(
            out_args, out_types, shape, self.out_params, is_size_specified)
        if self.no_return:
            ret = None
        elif not self.return_tuple and self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if _contains_zero(shape):
            return ret

        for i, x in enumerate(in_args):
            if type(x) is _scalar.CScalar:
                (<_scalar.CScalar>x).apply_dtype(in_types[i])

        inout_args = in_args + out_args

        if self.reduce_dims:
            shape = _reduce_dims(inout_args, self.params, shape)
        indexer = _carray._indexer_init(shape)
        inout_args.append(indexer)

        arginfos = _get_arginfos(inout_args)
        kern = self._get_elementwise_kernel(dev_id, arginfos, type_map)
        kern.linear_launch(indexer.size, inout_args, shared_mem=0,
                           block_max_size=block_size, stream=stream)
        return ret

    cpdef tuple _decide_params_type(
            self, tuple in_args_dtype, tuple out_args_dtype):
        key = (in_args_dtype, out_args_dtype)
        ret = self._params_type_memo.get(key, None)
        if ret is not None:
            return ret
        ret = _decide_params_type_core(
            self.in_params, self.out_params, in_args_dtype, out_args_dtype)
        self._params_type_memo[key] = ret
        return ret

    cpdef function.Function _get_elementwise_kernel(
            self, int dev_id, tuple arginfos, _TypeMap type_map):
        key = (
            dev_id,
            arginfos,
            type_map)
        kern = self._elementwise_kernel_memo.get(key, None)
        if kern is not None:
            return kern
        kern = _get_elementwise_kernel(
            arginfos, type_map, self.params, self.operation,
            self.name, self.preamble, **self.kwargs)

        # Store the compiled kernel in the cache.
        # Potentially overwrite a duplicate cache entry because
        # _get_elementwise_kernel() may include IO wait.
        in_types = []
        for x in arginfos:
            if x.type is cupy.ndarray:
                in_types.append(cupy.dtype(x.dtype).char)
        in_types = tuple(in_types)
        if in_types not in self._cached_codes:
            code = _get_elementwise_kernel_code(
                arginfos, type_map, self.params, self.operation,
                self.name, self.preamble, **self.kwargs)
            self._cached_codes[in_types] = code
        self._elementwise_kernel_memo[key] = kern
        return kern

    @property
    def cached_codes(self):
        """Returns a dict that has input types as keys and codes values.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
        if len(self._cached_codes) == 0:
            warnings.warn(
                'No codes are cached because compilation is deferred until '
                'the first function call.')
        return dict([(k, v) for k, v in self._cached_codes.items()])

    @property
    def cached_code(self):
        """Returns `next(iter(self.cached_codes.values()))`.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
        codes = self._cached_codes
        if len(codes) > 1:
            warnings.warn(
                'The input types of the kernel could not be inferred. '
                'Please use `.cached_codes` instead.')
        return next(iter(codes.values()))


cdef str fix_cast_expr(src_type, dst_type, str expr):
    src_kind = get_dtype(src_type).kind
    dst_kind = get_dtype(dst_type).kind
    if src_kind == dst_kind:
        return expr
    if src_kind == 'b':
        # HIP has an issue with bool conversions detailed below
        if runtime._is_hip_environment:
            return f'_hip_bool_cast({expr})'
        else:
            return f'({expr}) ? 1 : 0'
    if src_kind == 'c':
        if dst_kind == 'b':
            return f'({expr}) != {_scalar.get_typename(src_type)}()'
        else:  # dst_kind in 'iuf' (int, uint, float)
            return f'({expr}).real()'
    return expr


cdef function.Function _get_ufunc_kernel(
        tuple in_types, tuple out_types, routine, tuple arginfos,
        bint has_where, params,
        name, preamble, loop_prep):
    cdef _ArgInfo arginfo
    cdef str str_type, str_var

    offset_where = len(in_types)
    offset_out = offset_where
    if has_where:
        offset_out += 1

    types = []
    op = []
    if has_where:
        arginfo = arginfos[offset_where]
        if arginfo.is_ndarray():
            op.append('if(!_raw__where[_ind.get()]) continue;')
        else:
            op.append('if(!_where) continue;')
    for i, x in enumerate(in_types):
        str_var = 'in%d' % i
        str_type = str_var + '_type'
        types.append((str_type, x))
        arginfo = arginfos[i]
        if arginfo.is_ndarray():
            op.append('const {} {}({});'.format(
                str_type,
                str_var,
                fix_cast_expr(arginfo.dtype, x, f'_raw_{str_var}[_ind.get()]')
            ))

    out_op = []
    for i, x in enumerate(out_types):
        str_var = 'out%d' % i
        str_type = str_var + '_type'
        types.append((str_type, x))
        arginfo = arginfos[i + offset_out]
        op.append(f'{str_type} {str_var};')
        out_op.append('{} = {};'.format(
            f'_raw_{str_var}[_ind.get()]',
            fix_cast_expr(x, arginfo.dtype, str_var)
        ))

    type_map = _TypeMap(tuple(types))

    op.append(routine)
    op.append(';')
    op.extend(out_op)
    operation = '\n'.join(op)
    # HIP/ROCm 4.3 has an issue with ifs and ternary operators
    #
    # int bool(int x) {
    #     if (x != 0) return 1;
    #     return 0;
    # }
    #
    # bool(5) == 1;  //false
    # bool(5) == 5;  //true
    #
    # also it simplifies  (a ? 1 : 0)  directly to a, and yields
    # an incorrect value
    if runtime._is_hip_environment:
        preamble += """
        __device__ int _hip_bool_cast(long long int x) {
            volatile int a = 1;
            if (x == 0) a = 0;
            return a;
        }
        """
    return _get_simple_elementwise_kernel(
        params, arginfos, operation, name, type_map, preamble,
        loop_prep=loop_prep)


cdef inline bint _check_should_use_min_scalar(list in_args) except? -1:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for i in in_args:
        kind = _get_kind_score(ord(i.dtype.kind))
        if isinstance(i, _ndarray_base):
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            not all_scalars and
            max_array_kind >= max_scalar_kind)


cdef dict _mst_unsigned_to_signed = {
    i: (numpy.iinfo(j).max, (i, j))
    for i, j in [(numpy.dtype(i).type, numpy.dtype(i.lower()).type)
                 for i in "BHILQ"]}
cdef _numpy_min_scalar_type = numpy.min_scalar_type

cdef _min_scalar_type(x):
    # A non-negative integer may have two locally minimum scalar
    # types: signed/unsigned integer.
    # Return both for can_cast, while numpy.min_scalar_type only returns
    # the unsigned type.
    t = _numpy_min_scalar_type(x)
    dt = t.type
    if t.kind == 'u':
        m, dt2 = <tuple>_mst_unsigned_to_signed[dt]
        if x <= m:
            return dt2
    return dt


cdef class ufunc:

    """Universal function.

    Attributes:
        ~ufunc.name (str): The name of the universal function.
        ~ufunc.nin (int): Number of input arguments.
        ~ufunc.nout (int): Number of output arguments.
        ~ufunc.nargs (int): Number of all arguments.

    """

    cdef:
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly object name
        readonly _Ops _ops  # normal routines
        # routines based on explicitly given output dtype
        readonly _Ops _out_ops
        readonly object _preamble
        readonly object _loop_prep
        readonly object _default_casting
        readonly object _cutensor_op
        readonly int _cutensor_alpha
        readonly int _cutensor_gamma
        readonly str _scatter_op
        readonly tuple _params
        readonly tuple _params_with_where
        readonly dict _routine_cache
        readonly dict _kernel_memo
        readonly object _doc
        public object __doc__
        readonly object __name__
        readonly object __module__

    def __init__(
            self, name, nin, nout, _Ops ops, preamble='', loop_prep='', doc='',
            default_casting=None, *, _Ops out_ops=None, cutensor_op=None,
            scatter_op=None):
        self.name = name
        self.__name__ = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        self._out_ops = out_ops
        self._preamble = preamble
        self._loop_prep = loop_prep
        self._doc = doc
        self.__doc__ = doc
        if default_casting is None:
            self._default_casting = 'same_kind'
        else:
            self._default_casting = default_casting

        if cutensor_op is not None and cuda_cutensor is not None:
            self._cutensor_op, self._cutensor_alpha, self._cutensor_gamma = (
                getattr(cuda_cutensor, cutensor_op[0]),
                cutensor_op[1], cutensor_op[2])
        self._scatter_op = scatter_op

        _in_params = tuple(
            ParameterInfo('T in%d' % i, True)
            for i in range(nin))
        _out_params = tuple(
            ParameterInfo('T out%d' % i, False)
            for i in range(nout))
        _other_params = (
            ParameterInfo('CIndexer _ind', False),)
        self._params = _in_params + _out_params + _other_params
        self._params_with_where = (
            _in_params + (ParameterInfo('T _where', False),)
            + _out_params + _other_params)
        self._routine_cache = {}
        self._kernel_memo = {}

    def __repr__(self):
        return '<ufunc \'%s\'>' % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for op in self._ops.ops:
            in_str = ''.join([<str>get_dtype(t).char for t in op.in_types])
            out_str = ''.join([<str>get_dtype(t).char for t in op.out_types])
            types.append('%s->%s' % (in_str, out_str))
        return types

    def __call__(self, *args, **kwargs):
        """Applies the universal function to arguments elementwise.

        Args:
            args: Input arguments. Each of them can be a :class:`cupy.ndarray`
                object or a scalar. The output arguments can be omitted or be
                specified by the ``out`` argument.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.
            dtype: Data type specifier.

        Returns:
            Output array or a tuple of output arrays.

        """
        for arg in args:
            if hasattr(arg, '__cupy_override_elementwise_kernel__'):
                return arg.__cupy_override_elementwise_kernel__(
                    self, *args, **kwargs)

        if _fusion_thread_local.is_fusing():
            return _fusion_thread_local.call_ufunc(self, *args, **kwargs)

        cdef function.Function kern
        cdef list broad_values
        cdef shape_t shape

        out = kwargs.pop('out', None)
        where = kwargs.pop('_where', None)
        cdef bint has_where = where is not None
        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', self._default_casting)
        if dtype is not None:
            dtype = get_dtype(dtype).type
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if not (self.nin <= n_args <= self.nargs):
            # TODO(kataoka): Fix error message for nout >= 2 (e.g. divmod)
            raise TypeError(
                'Wrong number of arguments for {!r}. '
                'It must be either {} or {} (with outputs), '
                'but given {}.'.format(
                    self.name, self.nin, self.nargs, n_args))

        # parse inputs (positional) and outputs (positional or keyword)
        in_args = args[:self.nin]
        out_args = args[self.nin:]
        if out is not None:
            if out_args:
                raise ValueError('Cannot specify \'out\' as both '
                                 'a positional and keyword argument')
            if isinstance(out, tuple):
                if len(out) != self.nout:
                    raise ValueError(
                        "The 'out' tuple must have exactly one entry per "
                        "ufunc output")
                out_args = out
            else:
                if 1 != self.nout:
                    raise ValueError("'out' must be a tuple of arrays")
                out_args = out,

        dev_id = device.get_device_id()
        in_args = _preprocess_args(dev_id, in_args, False)
        out_args = _preprocess_optional_args(dev_id, out_args, False)
        given_out_args = [o for o in out_args if o is not None]

        # TODO(kataoka): Typecheck `in_args` w.r.t. `casting` (before
        # broadcast).
        if has_where:
            where_args = _preprocess_args(dev_id, (where,), False)
            x = where_args[0]
            if isinstance(x, _ndarray_base):
                # NumPy seems using casting=safe here
                if x.dtype != bool:
                    raise TypeError(
                        f'Cannot cast array data from {x.dtype!r} to '
                        f'{get_dtype(bool)!r} according to the rule \'safe\'')
            else:
                # NumPy does not seem raising TypeError.
                # CuPy does not have to support `where=object()` etc. and
                # `_preprocess_args` rejects it anyway.
                where_args[0] = _scalar.CScalar.from_numpy_scalar_with_dtype(
                    x, numpy.bool_)
        else:
            where_args = []

        # _copy_in_args_if_needed updates in_args
        _copy_in_args_if_needed(in_args, given_out_args)
        _copy_in_args_if_needed(where_args, given_out_args)
        broad_values = in_args + where_args + given_out_args
        # _broadcast updates shape
        internal._broadcast_core(broad_values, shape)

        if (self._cutensor_op is not None
                and _accelerator.ACCELERATOR_CUTENSOR in
                _accelerator._elementwise_accelerators):
            if (self.nin == 2 and self.nout == 1 and
                    isinstance(in_args[0], _ndarray_base) and
                    isinstance(in_args[1], _ndarray_base)):
                import cupyx.cutensor
                ret = cupyx.cutensor._try_elementwise_binary_routine(
                    in_args[0], in_args[1], dtype,
                    out_args[0] if len(out_args) == 1 else None,
                    self._cutensor_op,
                    self._cutensor_alpha,
                    self._cutensor_gamma,
                )
                if ret is not None:
                    return ret

        op = self._ops.guess_routine(
            self.name, self._routine_cache, in_args, dtype, self._out_ops)

        # Determine a template object from which we initialize the output when
        # inputs have subclass instances
        def issubclass1(cls, classinfo):
            return issubclass(cls, classinfo) and cls is not classinfo
        subtype = cupy.ndarray
        template = None
        for in_arg in in_args:
            in_arg_type = type(in_arg)
            if issubclass1(in_arg_type, cupy.ndarray):
                subtype = in_arg_type
                template = in_arg
                break

        out_args = _get_out_args_from_optionals(
            subtype, out_args, op.out_types, shape, casting, template)
        if self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if _contains_zero(shape):
            return ret

        inout_args = []
        for i, t in enumerate(op.in_types):
            x = broad_values[i]
            inout_args.append(
                x if isinstance(x, _ndarray_base) else
                _scalar.CScalar.from_numpy_scalar_with_dtype(x, t))
        if has_where:
            x = broad_values[self.nin]
            inout_args.append(x)
        inout_args.extend(out_args)
        shape = _reduce_dims(inout_args, self._params, shape)
        indexer = _carray._indexer_init(shape)
        inout_args.append(indexer)
        arginfos = _get_arginfos(inout_args)

        kern = self._get_ufunc_kernel(dev_id, op, arginfos, has_where)

        kern.linear_launch(indexer.size, inout_args)
        return ret

    cdef str _get_name_with_type(self, tuple arginfos, bint has_where):
        cdef str name = self.name
        if has_where:
            name += '_where'
        cdef _ArgInfo arginfo
        inout_type_words = []
        for arginfo in arginfos:
            dtype = str(numpy.dtype(arginfo.dtype))
            if arginfo.is_ndarray():
                inout_type_words.append(dtype)
            elif arginfo.is_scalar():
                inout_type_words.append(dtype.rstrip('0123456789'))
        return '{}__{}'.format(name, '_'.join(inout_type_words))

    cdef function.Function _get_ufunc_kernel(
            self, int dev_id, _Op op, tuple arginfos, bint has_where):
        cdef function.Function kern
        key = (dev_id, op, arginfos, has_where)
        kern = self._kernel_memo.get(key, None)
        if kern is None:
            name = self._get_name_with_type(arginfos, has_where)
            params = self._params_with_where if has_where else self._params
            kern = _get_ufunc_kernel(
                op.in_types, op.out_types, op.routine, arginfos, has_where,
                params, name, self._preamble, self._loop_prep)
            self._kernel_memo[key] = kern
        return kern

    def outer(self, A, B, **kwargs):
        """Apply the ufunc operation to all pairs of elements in A and B.

        .. seealso::
           :meth:`numpy.ufunc.outer`

        """
        A = core.array(A)
        B = core.array(B)
        ndim_a = A.ndim
        ndim_b = B.ndim
        A = A.reshape(A.shape + (1,) * ndim_b)
        B = B.reshape((1,) * ndim_a + B.shape)
        return self(A, B, **kwargs)

    def at(self, a, indices, b=None):
        """Apply in place operation on the operand ``a`` for elements
        specified by ``indices``.

        .. seealso::
           :meth:`numpy.ufunc.at`
        """
        if self._scatter_op is not None:
            a._scatter_op(indices, b, self._scatter_op)
        else:
            raise NotImplementedError(f'`{self.name}.at` is not supported yet')

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False):
        """Reduce ``array`` applying ufunc.

        .. seealso::
           :meth:`numpy.ufunc.reduce`
        """
        if self.name == 'cupy_add':
            return array.sum(axis, dtype, out, keepdims)
        if self.name == 'cupy_multiply':
            return array.prod(axis, dtype, out, keepdims)
        raise NotImplementedError(f'`{self.name}.reduce` is not supported yet')

    def accumulate(self, array, axis=0, dtype=None, out=None):
        """Accumulate ``array`` applying ufunc.

        .. seealso::
           :meth:`numpy.ufunc.accumulate`
        """
        if self.name == 'cupy_add':
            return array.cumsum(axis, dtype, out)
        if self.name == 'cupy_multiply':
            return array.cumprod(axis, dtype, out)
        raise NotImplementedError(
            f'`{self.name}.accumulate` is not supported yet')

    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        """Reduce ``array`` applying ufunc with indices.

        .. seealso::
           :meth:`numpy.ufunc.reduceat`
        """
        if self.name == 'cupy_add':
            return array._add_reduceat(indices, axis, dtype, out)
        raise NotImplementedError(
            f'`{self.name}.reduceat` is not supported yet')


def _ufunc_doc_signature_formatter(ufunc, name):
    # Based on implementation in NumPy (numpy/_core/_internal.py)

    # input arguments are simple
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        in_args = ', '.join(f'x{i+1}' for i in range(ufunc.nin))

    # output arguments are both keyword or positional
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        out_args = '[, {positional}], / [, out={default}]'.format(
            positional=', '.join(
                'out{}'.format(i+1) for i in range(ufunc.nout)),
            default=repr((None,)*ufunc.nout)
        )

    # keyword only args depend on whether this is a gufunc
    kwargs = (
        f", casting='{ufunc._default_casting}'"
        ", dtype=None"
    )

    # join all the parts together
    return r'{name}({in_args}{out_args}, \*{kwargs})'.format(
        name=name,
        in_args=in_args,
        out_args=out_args,
        kwargs=kwargs
    )


cdef class _Op:

    def __init__(
            self, tuple in_types, tuple out_types, object routine,
            object error_func):
        if error_func is None:
            assert routine is not None
        else:
            assert callable(error_func)
        self.in_types = in_types
        self.out_types = out_types
        self.nin = len(in_types)
        self.nout = len(out_types)
        self.routine = routine
        self.error_func = error_func

    @staticmethod
    cdef _Op _from_type_and_routine_or_error_func(
            str typ, object routine, object error_func):
        # TODO(niboshi): Write type mapping specification.
        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([get_dtype(t).type for t in in_types])
        out_types = tuple([get_dtype(t).type for t in out_types])
        return _Op(in_types, out_types, routine, error_func)

    @staticmethod
    cdef _Op from_type_and_routine(str typ, routine):
        return _Op._from_type_and_routine_or_error_func(typ, routine, None)

    @staticmethod
    cdef _Op from_type_and_error_func(str typ, error_func):
        return _Op._from_type_and_routine_or_error_func(typ, None, error_func)

    cdef check_valid(self):
        if self.error_func is not None:
            self.error_func()

    cpdef tuple get_in_dtypes(self):
        return tuple([get_dtype(t) for t in self.in_types])

    cpdef tuple get_out_dtypes(self):
        return tuple([get_dtype(t) for t in self.out_types])


cdef class _Ops:

    def __init__(self, tuple ops):
        assert len(ops) > 0
        nin = ops[0].nin
        nout = ops[0].nout
        assert all(op.nin == nin for op in ops)
        assert all(op.nout == nout for op in ops)
        self.ops = ops
        self.nin = nin
        self.nout = nout

    @staticmethod
    cdef _Ops from_tuples(object ops, routine):
        ops_ = []
        for t in ops:
            if isinstance(t, tuple):
                typ, rt = t
                if isinstance(rt, tuple):
                    rt = tuple([r1 or r2 for r1, r2 in zip(rt, routine)])
                elif not isinstance(rt, str):
                    assert callable(rt)
                    ops_.append(_Op.from_type_and_error_func(typ, rt))
                    continue
            else:
                assert isinstance(t, str)
                typ, rt = t, routine
            ops_.append(_Op.from_type_and_routine(typ, rt))
        return _Ops(tuple(ops_))

    cpdef _Op guess_routine(
            self, str name, dict cache, list in_args, dtype, _Ops out_ops):
        cdef _Ops ops_
        if dtype is None:
            use_raw_value = _check_should_use_min_scalar(in_args)
            if use_raw_value:
                in_types = tuple([
                    a.dtype.type if isinstance(a, _ndarray_base)
                    else _min_scalar_type(a)
                    for a in in_args])
            else:
                in_types = tuple([a.dtype.type for a in in_args])
            op = cache.get(in_types, ())
            if op is ():
                op = self._guess_routine_from_in_types(in_types)
                cache[in_types] = op
        else:
            op = cache.get(dtype, ())
            if op is ():
                ops_ = out_ops or self
                op = ops_._guess_routine_from_dtype(dtype)
                cache[dtype] = op

        if op is not None:
            # raise TypeError if the type combination is disallowed
            (<_Op>op).check_valid()

            return op

        if dtype is None:
            dtype = tuple([a.dtype.type for a in in_args])
        raise TypeError('Wrong type (%s) of arguments for %s' %
                        (dtype, name))

    cpdef _Op _guess_routine_from_in_types(
            self, tuple in_types, object can_cast=_numpy_can_cast):
        cdef _Op op
        cdef tuple op_types
        cdef Py_ssize_t n = len(in_types)
        cdef Py_ssize_t i
        for op in self.ops:
            op_types = op.in_types
            for i in range(n):
                it = in_types[i]
                ot = op_types[i]
                if isinstance(it, tuple):
                    if not can_cast(it[0], ot) and not can_cast(it[1], ot):
                        break
                elif not can_cast(it, ot):
                    break
            else:
                return op
        return None

    cpdef _Op _guess_routine_from_dtype(self, object dtype):
        cdef _Op op
        cdef tuple op_types
        for op in self.ops:
            op_types = op.out_types
            for t in op_types:
                if t != dtype:
                    break
            else:
                return op
        return None


cpdef create_ufunc(name, ops, routine=None, preamble='', doc='',
                   default_casting=None, loop_prep='', out_ops=None,
                   cutensor_op=None, scatter_op=None):
    ops_ = _Ops.from_tuples(ops, routine)
    _out_ops = None if out_ops is None else _Ops.from_tuples(out_ops, routine)
    return ufunc(
        name, ops_.nin, ops_.nout, ops_, preamble,
        loop_prep, doc, default_casting=default_casting, out_ops=_out_ops,
        cutensor_op=cutensor_op, scatter_op=scatter_op)

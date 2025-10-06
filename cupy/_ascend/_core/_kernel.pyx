import string
import warnings

import numpy

import cupy
from cupy import _util

cimport cython  # NOQA

from libcpp cimport vector
from cupy.cuda cimport device
from cupy.cuda cimport memory
from cupy._core cimport _carray
from cupy._core cimport _scalar
from cupy._core._dtype cimport get_dtype, _raise_if_invalid_cast
from cupy._core._memory_range cimport may_share_bounds
from cupy._core._scalar import get_typename as _get_typename
from cupy._core cimport core

from cupy._core._routines_creation cimport _ndarray_init
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal
from cupy_backends.cuda.api cimport runtime

cdef inline bint _contains_zero(const shape_t& v) except? -1:
    for i in range(v.size()):
        if v[i] == 0:
            return True
    return False

@_util.memoize(for_each_device=True)
def _get_warpsize():
    device_id = runtime.getDevice()
    # TODO: ASCEND not compatible,  return 32 typical for NV gpu
    #return runtime.getDeviceProperties(device_id)['warpSize']
    return 32

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
    weak_t = False
    if isinstance(arg, _ndarray_base):
        s = arg
        _check_peer_access(<_ndarray_base>s, dev_id)
    elif hasattr(arg, '__cupy_get_ndarray__'):
        s = arg.__cupy_get_ndarray__()
        _check_peer_access(<_ndarray_base>s, dev_id)
    else:  # scalars or invalid args
        weak_t = type(arg) if type(arg) in [int, float, complex] else False
        if use_c_scalar:
            s = _scalar.scalar_to_c_scalar(arg)
        else:
            s = _scalar.scalar_to_numpy_scalar(arg)
        if s is None:
            raise TypeError('Unsupported type %s' % type(arg))
    return s, weak_t


cdef tuple _preprocess_args(int dev_id, args, bint use_c_scalar):
    """Preprocesses arguments for kernel invocation

    - Checks device compatibility for ndarrays
    - Converts Python/NumPy scalars:
      - If use_c_scalar is True, into CScalars.
      - If use_c_scalar is False, into NumPy scalars.
    """
    cdef list ret = []
    cdef list weaks = []
    for arg in args:
        p_arg, weak_t = _preprocess_arg(dev_id, arg, use_c_scalar)
        ret.append(p_arg)
        weaks.append(weak_t)
    return ret, tuple(weaks)


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
            ret.append(_preprocess_arg(dev_id, arg, use_c_scalar)[0])
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
        IF CUPY_CANN_VERSION <= 0:
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

    IF CUPY_CANN_VERSION <= 0:
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
        IF CUPY_CANN_VERSION <= 0:
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
cdef _numpy_result_type = numpy.result_type

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



cdef dict _mst_unsigned_to_signed = {
    i: (numpy.iinfo(j).max, (i, j))
    for i, j in [(numpy.dtype(i).type, numpy.dtype(i.lower()).type)
                 for i in "BHILQ"]}


cdef inline int _get_kind_score(type kind):
    if issubclass(kind, numpy.bool_):
        return 0
    if issubclass(kind, (numpy.integer, int)):
        return 1
    if issubclass(kind, (numpy.inexact, float, complex)):
        return 2
    # unknown type, assume higher score
    return 3


cdef inline bint _check_should_use_weak_scalar(
    tuple in_types, tuple weaks
) except? -1:
    """The promotion strategy of finding the first matching loop is not
    equipped to deal with correct promotion when mixing weak scalars and
    arrays/strong types.
    To fix this we (also NumPy when it uses this strategy) check if the scalars
    have a higher kind and do not use them if they do.

    This prevents e.g. `uint8(1) + 0.` from picking a float64 loop.
    """
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars_or_arrays

    if weaks is None:
        # equivalent to (False,)*len(in_types)
        return False

    max_array_kind = -1
    max_scalar_kind = -1
    for in_t, w_t in zip(in_types, weaks):
        if w_t:
            kind = _get_kind_score(w_t)
            max_scalar_kind = max(max_scalar_kind, kind)
        else:
            kind = _get_kind_score(in_t)
            max_array_kind = max(max_array_kind, kind)

    all_scalars_or_arrays = max_scalar_kind == -1 or max_array_kind == -1

    return not all_scalars_or_arrays and max_array_kind >= max_scalar_kind


cdef class ufunc:
    """Universal function (simplifed for ascend aclnnop), 
    remove: where_param, cutensor, fusion, function.Function
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
        readonly object _default_casting
        readonly str _scatter_op
        readonly tuple _params
        readonly object _doc
        public object __doc__
        readonly object __name__

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
        self._doc = doc
        self.__doc__ = doc
        if default_casting is None:
            self._default_casting = 'same_kind'
        else:
            self._default_casting = default_casting
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
        in_args, weaks = _preprocess_args(dev_id, in_args, False)
        out_args = _preprocess_optional_args(dev_id, out_args, False)
        given_out_args = [o for o in out_args if o is not None]

        # _copy_in_args_if_needed updates in_args
        _copy_in_args_if_needed(in_args, given_out_args)
        #_copy_in_args_if_needed(where_args, given_out_args)
        broad_values = in_args + given_out_args
        # _broadcast updates shape
        internal._broadcast_core(broad_values, shape)

        op = self._ops.guess_routine(
            self.name, self._routine_cache, in_args, weaks, dtype,
            self._out_ops)

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

        # TODO: ASCEND launch kernel
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
            self, str name, dict cache, list in_args, tuple weaks, dtype,
            _Ops out_ops):
        cdef _Ops ops_

        if dtype is None:
            assert all([isinstance(a, (_ndarray_base, numpy.generic))
                        for a in in_args])
            in_types = tuple([a.dtype.type for a in in_args])

            if not _check_should_use_weak_scalar(in_types, weaks):
                weaks = (False,) * len(in_args)

            op = cache.get((in_types, weaks), ())
            if op is ():
                op = self._guess_routine_from_in_types(in_types, weaks)
                cache[(in_types, weaks)] = op
        else:
            op = cache.get(dtype, ())
            if op is ():
                ops_ = out_ops or self
                op = ops_._guess_routine_from_dtype(dtype)
                cache[dtype] = op

        if op is not None:
            # raise TypeError if the type combination is disallowed
            (<_Op>op).check_valid()

            if weaks is None:
                return op

            # check for overflow in operands. Consider `np.uint8(1) + 300`.
            # Per NEP 50 this raises OverflowError because 300 overflows uint8.
            # We can only check it after the operand types are known
            for i in range(len(in_args)):
                if weaks[i] is not int:
                    continue
                # Note: For simplicity, check even if `in_type` is e.g. float
                integer_argument = int(in_args[i])
                in_type = op.in_types[i]
                in_type(integer_argument)  # Check if user input fits loop

            return op

        if dtype is None:
            dtype = tuple([a.dtype.type for a in in_args])
        raise TypeError('Wrong type (%s) of arguments for %s' %
                        (dtype, name))

    cpdef _Op _guess_routine_from_in_types(
            self, tuple in_types, tuple weaks=None,
            object can_cast=_numpy_can_cast
    ):
        cdef _Op op
        cdef tuple op_types
        cdef Py_ssize_t n = len(in_types)
        cdef Py_ssize_t i

        for op in self.ops:
            op_types = op.in_types
            for i in range(n):
                it = in_types[i]
                ot = op_types[i]
                weak_t = weaks[i] if weaks is not None else False

                # XXX: Remove assert (reachable only for pre NEP 50 logic)
                assert(not isinstance(it, tuple))

                if not can_cast(it, ot):
                    if not weak_t:
                        break

                    # If `result_type` doesn't return `ot` then the weak
                    # scalar caused promotion and operand cannot be used.
                    try:
                        if _numpy_result_type(weak_t(0), ot) != ot:
                            break
                    except TypeError:
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

# ASCEND refactor:  TODO **kwargs arg how??
# emulate ElementwiseKernel class with create_ufunc()        
cpdef ElementwiseKernel(in_params, out_params, operation,
                 name='my_kernel_name', reduce_dims=True, preamble='',
                 no_return=False, return_tuple=False, doc=""):
    ops_ = _ops_from_tuples((in_params, out_params), operation)
    return ufunc(
        name, ops_.nin, ops_.nout, ops_, preamble,
        doc, default_casting=default_casting, out_ops=None,
        cutensor_op=None, scatter_op=None)


# TODO: ASCEND not impl yet
cdef _ops_from_tuples(object ops, routine):
    ops_ = []
    #raise NotImplementedError("ElementwiseKernel() not yet impl")  # TODO: import time error
    """
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
    """
    return _Ops(tuple(ops_))

# TODO: ASCEND not impl yet
cpdef create_reduction_func(
        name, ops, routine=None, identity=None, preamble='',
        sort_reduce_axis=True):
    ops_ = _Ops.from_tuples(ops, routine)
    return ufunc(
        name, ops_.nin, ops_.nout, ops_, preamble)
# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
cimport cython  # NOQA
cimport cpython
from libc.stdint cimport int64_t, intptr_t

import os
import numpy
import warnings

import cupy
from cupy import _util
from cupy._core._ufuncs import elementwise_copy
from cupy._core.core cimport _ndarray_base
from cupy._core._ndarray import ndarray
from cupy._core.core cimport shape_t, strides_t
from cupy._core cimport _dtype
from cupy._core._dtype cimport get_dtype
from cupy._core cimport internal
from cupy._core cimport _routines_manipulation as _manipulation

from cupy.xpu import memory as memory_module
from cupy.xpu cimport stream as stream_module
from cupy.xpu cimport device
from cupy.xpu cimport pinned_memory
from cupy.xpu cimport memory

from backends.backend.api cimport runtime
from backends.backend.api.runtime import CUDARuntimeError


NUMPY_1x = numpy.__version__ < '2'

cdef bint _is_ump_enabled = (int(os.environ.get('CUPY_ENABLE_UMP', '0')) != 0)

cdef inline bint is_ump_supported(int device_id) except*:
    IF CUPY_CANN_VERSION <= 0:
        if (_is_ump_enabled
                # 1 for both HMM/ATS addressing modes
                # this assumes device_id is a GPU device ordinal (not -1)
                and runtime.deviceGetAttribute(
                    runtime.cudaDevAttrPageableMemoryAccess, device_id)):
            return True
        else:
            return False
    ELSE:
        return False

cpdef _ndarray_base array(obj, dtype=None, copy=True, order='K',
                          bint subok=False, Py_ssize_t ndmin=0,
                          bint blocking=False):
    # TODO(beam2d): Support subok options
    if subok:
        raise NotImplementedError
    if order is None:
        order = 'K'

    if isinstance(obj, ndarray):
        return _array_from_cupy_ndarray(obj, dtype, copy, order, ndmin)

    if hasattr(obj, '__cuda_array_interface__'):
        return _array_from_cuda_array_interface(
            obj, dtype, copy, order, subok, ndmin)

    if hasattr(obj, '__cupy_get_ndarray__'):
        return _array_from_cupy_ndarray(
            obj.__cupy_get_ndarray__(), dtype, copy, order, ndmin)

    concat_shape, concat_type, concat_dtype = (
        _array_info_from_nested_sequence(obj))
    if concat_shape is not None:
        return _array_from_nested_sequence(
            obj, dtype, order, ndmin, concat_shape, concat_type, concat_dtype,
            blocking)

    return _array_default(obj, dtype, copy, order, ndmin, blocking)


cdef _ndarray_base _array_from_cupy_ndarray(
        obj, dtype, copy, order, Py_ssize_t ndmin):
    cdef Py_ssize_t ndim
    cdef _ndarray_base a, src

    src = obj

    if dtype is None:
        dtype = src.dtype

    if src.data.device_id == device.get_device_id():
        a = src._astype(dtype, order=order, casting=None, subok=None, copy=copy)
    else:
        a = src.copy(order=order)._astype(dtype, order=order, casting=None, subok=None, copy=None)

    ndim = a._shape.size()
    if ndmin > ndim:
        if a is obj:
            # When `copy` is False, `a` is same as `obj`.
            a = a.view()
        a.shape = (1,) * (ndmin - ndim) + a.shape

    return a


cdef _ndarray_base _array_from_cuda_array_interface(
        obj, dtype, copy, order, bint subok, Py_ssize_t ndmin):
    return array(
        _convert_object_with_cuda_array_interface(obj),
        dtype, copy, order, subok, ndmin)


cdef _ndarray_base _array_from_nested_sequence(
        obj, dtype, order, Py_ssize_t ndmin, concat_shape, concat_type,
        concat_dtype, bint blocking):
    cdef Py_ssize_t ndim

    # resulting array is C order unless 'F' is explicitly specified
    # (i.e., it ignores order of element arrays in the sequence)
    order = (
        'F'
        if order is not None and len(order) >= 1 and order[0] in 'Ff'
        else 'C')

    ndim = len(concat_shape)
    if ndmin > ndim:
        concat_shape = (1,) * (ndmin - ndim) + concat_shape

    if dtype is None:
        dtype = concat_dtype.newbyteorder('<')

    if concat_type is numpy.ndarray:
        return _array_from_nested_numpy_sequence(
            obj, concat_dtype, dtype, concat_shape, order, ndmin,
            blocking)
    elif concat_type is ndarray:  # TODO(takagi) Consider subclases
        return _array_from_nested_cupy_sequence(
            obj, dtype, concat_shape, order, blocking)
    else:
        assert False


cdef _ndarray_base _array_from_nested_numpy_sequence(
        arrays, src_dtype, dst_dtype, const shape_t& shape, order,
        Py_ssize_t ndmin, bint blocking):
    a_dtype = get_dtype(dst_dtype)  # convert to numpy.dtype
    if a_dtype.char not in _dtype.all_type_chars:
        raise ValueError('Unsupported dtype %s' % a_dtype)
    cdef _ndarray_base a  # allocate it after pinned memory is secured
    cdef size_t itemcount = internal.prod(shape)
    cdef size_t nbytes = itemcount * a_dtype.itemsize

    stream = stream_module.get_current_stream()
    # Note: even if arrays are already backed by pinned memory, we still need
    # to allocate an extra buffer and copy from it to avoid potential data
    # race, see the discussion here:
    # https://github.com/cupy/cupy/pull/5155#discussion_r621808782
    cdef pinned_memory.PinnedMemoryPointer mem = (
        _alloc_async_transfer_buffer(nbytes))
    if mem is not None:
        # write concatenated arrays to the pinned memory directly
        src_cpu = (
            numpy.frombuffer(mem, a_dtype, itemcount)
            .reshape(shape, order=order))
        _concatenate_numpy_array(
            [numpy.expand_dims(e, 0) for e in arrays],
            0,
            get_dtype(src_dtype),
            a_dtype,
            src_cpu)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host_async(mem.ptr, nbytes, stream)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        # fallback to numpy array and send it to GPU
        # Note: a_cpu.ndim is always >= 1
        a_cpu = numpy.array(arrays, dtype=a_dtype, copy=False, order=order,
                            ndmin=ndmin)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host_async(a_cpu.ctypes.data, nbytes, stream)

    if blocking:
        stream.synchronize()

    return a


cdef _ndarray_base _array_from_nested_cupy_sequence(
        obj, dtype, shape, order, bint blocking):
    lst = _flatten_list(obj)

    # convert each scalar (0-dim) ndarray to 1-dim
    lst = [cupy.expand_dims(x, 0) if x.ndim == 0 else x for x in lst]

    a = _manipulation.concatenate_method(lst, 0)
    a = a.reshape(shape)
    a = a.astype(dtype, order=order, copy=False)

    if blocking:
        stream = stream_module.get_current_stream()
        stream.synchronize()

    return a


cdef inline _ndarray_base _try_skip_h2d_copy(
        obj, dtype, bint copy, order, Py_ssize_t ndmin):
    if copy:
        return None

    if not is_ump_supported(device.get_device_id()):
        return None

    if not isinstance(obj, numpy.ndarray):
        return None

    # dtype should not change
    obj_dtype = obj.dtype
    if not (obj_dtype == get_dtype(dtype) if dtype is not None else True):
        return None

    # CuPy only supports numerical dtypes
    if obj_dtype.char not in _dtype.all_type_chars:
        return None

    # CUDA only supports little endianness
    if obj_dtype.byteorder not in ('|', '=', '<'):
        return None

    # strides and the requested order could mismatch
    obj_flags = obj.flags
    if not internal._is_layout_expected(
            obj_flags.c_contiguous, obj_flags.f_contiguous, order):
        return None

    cdef intptr_t ptr = obj.ctypes.data

    # NumPy 0-size arrays still have non-null pointers...
    cdef size_t nbytes = obj.nbytes
    if nbytes == 0:
        ptr = 0

    cdef Py_ssize_t ndim = obj.ndim
    cdef tuple shape = obj.shape
    cdef tuple strides = obj.strides
    if ndmin > ndim:
        # pad shape & strides
        shape = (1,) * (ndmin - ndim) + shape
        strides = (shape[0] * strides[0],) * (ndmin - ndim) + strides

    cdef memory.SystemMemory ext_mem = memory.SystemMemory.from_external(
        ptr, nbytes, obj)
    cdef memory.MemoryPointer memptr = memory.MemoryPointer(ext_mem, 0)
    return ndarray(shape, obj_dtype, memptr, strides)


cdef _ndarray_base _array_default(
        obj, dtype, copy, order, Py_ssize_t ndmin, bint blocking):
    cdef _ndarray_base a

    # Fast path: zero-copy a NumPy array if possible
    if not blocking:
        a = _try_skip_h2d_copy(obj, dtype, copy, order, ndmin)
        if a is not None:
            return a

    if order is not None and len(order) >= 1 and order[0] in 'KAka':
        if isinstance(obj, numpy.ndarray) and obj.flags.fnc:
            order = 'F'
        else:
            order = 'C'

    copy = False if NUMPY_1x else None

    a_cpu = numpy.array(obj, dtype=dtype, copy=copy, order=order,
                        ndmin=ndmin)
    if a_cpu.dtype.char not in _dtype.all_type_chars:
        raise ValueError('Unsupported dtype %s' % a_cpu.dtype)
    a_cpu = a_cpu.astype(a_cpu.dtype.newbyteorder('<'), copy=False)
    a_dtype = a_cpu.dtype

    # We already made a copy, we should be able to use it
    if _is_ump_enabled:
        a = _try_skip_h2d_copy(a_cpu, a_dtype, False, order, ndmin)
        assert a is not None
        return a

    cdef shape_t a_shape = a_cpu.shape
    a = ndarray(a_shape, dtype=a_dtype, order=order)
    if a_cpu.ndim == 0:
        a.fill(a_cpu)
        return a
    cdef Py_ssize_t nbytes = a.nbytes

    cdef pinned_memory.PinnedMemoryPointer mem
    stream = stream_module.get_current_stream()

    cdef intptr_t ptr_h = <intptr_t>(a_cpu.ctypes.data)
    if pinned_memory.is_memory_pinned(ptr_h):
        a.data.copy_from_host_async(ptr_h, nbytes, stream)
        pinned_memory._add_to_watch_list(stream.record(), a_cpu)
    else:
        # The input numpy array does not live on pinned memory, so we allocate
        # an extra buffer and copy from it to avoid potential data race, see
        # the discussion here:
        # https://github.com/cupy/cupy/pull/5155#discussion_r621808782
        mem = _alloc_async_transfer_buffer(nbytes)
        if mem is not None:
            src_cpu = numpy.frombuffer(mem, a_dtype, a_cpu.size)
            src_cpu[:] = a_cpu.ravel(order)
            a.data.copy_from_host_async(mem.ptr, nbytes, stream)
            pinned_memory._add_to_watch_list(stream.record(), mem)
        else:
            a.data.copy_from_host_async(ptr_h, nbytes, stream)

    if blocking:
        stream.synchronize()

    return a


cdef tuple _array_info_from_nested_sequence(obj):
    # Returns a tuple containing information if we can simply concatenate the
    # input to make a CuPy array (i.e., a (nested) sequence that only contains
    # NumPy/CuPy arrays with the same shape and dtype). `(None, None, None)`
    # means we do not concatenate the input.
    # 1. A concatenated shape
    # 2. The type of the arrays to concatenate (numpy.ndarray or cupy.ndarray)
    # 3. The dtype of the arrays to concatenate
    if isinstance(obj, (list, tuple)):
        return _compute_concat_info_impl(obj)
    else:
        return None, None, None


cdef tuple _compute_concat_info_impl(obj):
    cdef Py_ssize_t dim

    if isinstance(obj, (numpy.ndarray, ndarray)):
        return obj.shape, type(obj), obj.dtype

    if hasattr(obj, '__cupy_get_ndarray__'):
        return obj.shape, ndarray, obj.dtype

    if isinstance(obj, (list, tuple)):
        dim = len(obj)
        if dim == 0:
            return None, None, None

        concat_shape, concat_type, concat_dtype = (
            _compute_concat_info_impl(obj[0]))
        if concat_shape is None:
            return None, None, None

        for elem in obj[1:]:
            concat_shape1, concat_type1, concat_dtype1 = (
                _compute_concat_info_impl(elem))
            if concat_shape1 is None:
                return None, None, None

            if concat_shape != concat_shape1:
                return None, None, None
            if concat_type is not concat_type1:
                return None, None, None
            if concat_dtype != concat_dtype1:
                concat_dtype = numpy.promote_types(concat_dtype, concat_dtype1)

        return (dim,) + concat_shape, concat_type, concat_dtype

    return None, None, None


cdef list _flatten_list(object obj):
    ret = []
    if isinstance(obj, (list, tuple)):
        for elem in obj:
            ret += _flatten_list(elem)
        return ret
    return [obj]


cdef bint _numpy_concatenate_has_out_argument = (
    numpy.lib.NumpyVersion(numpy.__version__) >= '1.14.0')


cdef inline _concatenate_numpy_array(arrays, axis, src_dtype, dst_dtype, out):
    # type(*_dtype) must be numpy.dtype

    if (_numpy_concatenate_has_out_argument
            and src_dtype.kind == dst_dtype.kind):
        # concatenate only accepts same_kind casting
        numpy.concatenate(arrays, axis, out)
    else:
        out[:] = numpy.concatenate(arrays, axis)


cdef inline _alloc_async_transfer_buffer(Py_ssize_t nbytes):
    try:
        return pinned_memory.alloc_pinned_memory(nbytes)
    except CUDARuntimeError as e:
        if e.status != runtime.errorMemoryAllocation:
            raise
        warnings.warn(
            'Using synchronous transfer as pinned memory ({} bytes) '
            'could not be allocated. '
            'This generally occurs because of insufficient host memory. '
            'The original error was: {}'.format(nbytes, e),
            _util.PerformanceWarning)

    return None


cpdef _ndarray_base _internal_ascontiguousarray(_ndarray_base a):
    if a._c_contiguous:
        return a
    newarray = _ndarray_init(ndarray, a._shape, a.dtype, None)
    elementwise_copy(a, newarray)
    return newarray


cpdef _ndarray_base _internal_asfortranarray(_ndarray_base a):
    from backends.backend.libs import cublas

    cdef _ndarray_base newarray
    cdef int m, n
    cdef intptr_t handle

    if a._f_contiguous:
        return a

    newarray = ndarray(a.shape, a.dtype, order='F')
    if (a._c_contiguous and a._shape.size() == 2 and
            (a.dtype == numpy.float32 or a.dtype == numpy.float64)):
        m, n = a.shape
        handle = device.get_cublas_handle()
        one = numpy.array(1, dtype=a.dtype)
        zero = numpy.array(0, dtype=a.dtype)
        if a.dtype == numpy.float32:
            cublas.sgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, one.ctypes.data, a.data.ptr, n,
                zero.ctypes.data, a.data.ptr, n, newarray.data.ptr, m)
        elif a.dtype == numpy.float64:
            cublas.dgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, one.ctypes.data, a.data.ptr, n,
                zero.ctypes.data, a.data.ptr, n, newarray.data.ptr, m)
    else:
        elementwise_copy(a, newarray)
    return newarray


cpdef _ndarray_base ascontiguousarray(_ndarray_base a, dtype=None):
    cdef bint same_dtype = False
    zero_dim = a._shape.size() == 0
    if dtype is None:
        same_dtype = True
        dtype = a.dtype
    else:
        dtype = get_dtype(dtype)
        same_dtype = dtype == a.dtype

    if same_dtype and a._c_contiguous:
        if zero_dim:
            return _manipulation._ndarray_ravel(a, 'C')
        return a

    shape = (1,) if zero_dim else a.shape
    newarray = ndarray(shape, dtype)
    elementwise_copy(a, newarray)
    return newarray


cpdef _ndarray_base asfortranarray(_ndarray_base a, dtype=None):
    cdef _ndarray_base newarray
    cdef bint same_dtype = False
    zero_dim = a._shape.size() == 0

    if dtype is None:
        dtype = a.dtype
        same_dtype = True
    else:
        dtype = get_dtype(dtype)
        same_dtype = dtype == a.dtype

    if same_dtype and a._f_contiguous:
        if zero_dim:
            return _manipulation._ndarray_ravel(a, 'F')
        return a

    if same_dtype and not zero_dim:
        return _internal_asfortranarray(a)

    newarray = ndarray((1,) if zero_dim else a.shape, dtype, order='F')
    elementwise_copy(a, newarray)
    return newarray


cpdef _ndarray_base _convert_object_with_cuda_array_interface(a):
    if runtime._is_hip_environment:
        raise RuntimeError(
            'HIP/ROCm does not support cuda array interface')

    cdef Py_ssize_t sh, st
    cdef dict desc = a.__cuda_array_interface__
    cdef tuple shape = desc['shape']
    cdef int dev_id = -1
    cdef size_t nbytes

    ptr = desc['data'][0]
    dtype = numpy.dtype(desc['typestr'])
    if dtype.byteorder == '>':
        raise ValueError('CuPy does not support the big-endian byte-order')
    mask = desc.get('mask')
    if mask is not None:
        raise ValueError('CuPy currently does not support masked arrays.')
    strides = desc.get('strides')
    if strides is not None:
        nbytes = 0
        for sh, st in zip(shape, strides):
            nbytes = max(nbytes, abs(sh * st))
    else:
        nbytes = internal.prod_sequence(shape) * dtype.itemsize
    # the v2 protocol sets ptr=0 for 0-size arrays, so we can't look up
    # the pointer attributes and must use the current device
    if nbytes == 0:
        dev_id = device.get_device_id()
    mem = memory_module.UnownedMemory(ptr, nbytes, a, dev_id)
    memptr = memory.MemoryPointer(mem, 0)
    # the v3 protocol requires an immediate synchronization, unless
    # 1. the stream is not set (ex: from v0 ~ v2) or is None
    # 2. users explicitly overwrite this requirement
    stream_ptr = desc.get('stream')
    if stream_ptr is not None:
        if _util.CUDA_ARRAY_INTERFACE_SYNC:
            runtime.streamSynchronize(stream_ptr)
    return ndarray(shape, dtype, memptr, strides)


cdef _ndarray_base _ndarray_init(subtype, const shape_t& shape, dtype, obj):
    # Use `_no_init=True` for fast init. Now calling `__array_finalize__` is
    # responsibility of this function.
    cdef _ndarray_base ret = ndarray.__new__(subtype, _obj=obj, _no_init=True)
    ret._init_fast(shape, dtype, True)
    if subtype is not ndarray:
        ret.__array_finalize__(obj)
    return ret


cdef _ndarray_base _create_ndarray_from_shape_strides(
        subtype, const shape_t& shape, const strides_t& strides, dtype, obj):
    cdef int ndim = shape.size()
    cdef int64_t begin = 0, end = dtype.itemsize
    cdef memory.MemoryPointer ptr
    for i in range(ndim):
        if strides[i] > 0:
            end += strides[i] * (shape[i] - 1)
        elif strides[i] < 0:
            begin += strides[i] * (shape[i] - 1)
    ptr = memory.alloc(end - begin) + begin
    return ndarray.__new__(
        subtype, shape, dtype, _obj=obj, memptr=ptr, strides=strides)


cpdef min_scalar_type(a):
    """
    For scalar ``a``, returns the data type with the smallest size
    and smallest scalar kind which can hold its value.  For non-scalar
    array ``a``, returns the vector's dtype unmodified.

    .. seealso:: :func:`numpy.min_scalar_type`
    """
    if isinstance(a, ndarray):
        return a.dtype
    _, concat_type, concat_dtype = _array_info_from_nested_sequence(a)
    if concat_type is not None:
        return concat_dtype
    return numpy.min_scalar_type(a)

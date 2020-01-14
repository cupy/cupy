# distutils: language = c++
import ctypes
import warnings

import numpy

import cupy
from cupy.core import _ufuncs
from cupy.cuda import device
from cupy.cuda import memory as memory_module
from cupy.cuda import runtime
from cupy import util

cimport cython  # NOQA
from libcpp cimport vector

from cupy.core._kernel import ElementwiseKernel
from cupy.core cimport _dtype
from cupy.core cimport _routines_manipulation as _manipulation
from cupy.core cimport core
from cupy.core.core cimport ndarray
from cupy.cuda cimport cublas
from cupy.core cimport internal
from cupy.cuda cimport memory
from cupy.cuda cimport pinned_memory
from cupy.cuda cimport stream as stream_module


cpdef ndarray array(
        obj, dtype=None, bint copy=True, order='K', bint subok=False,
        Py_ssize_t ndmin=0):
    # TODO(beam2d): Support subok options
    cdef Py_ssize_t ndim
    cdef ndarray a, src
    cdef size_t nbytes

    if subok:
        raise NotImplementedError
    if order is None:
        order = 'K'

    if isinstance(obj, ndarray):
        src = obj
        if dtype is None:
            dtype = src.dtype
        if src.data.device_id == device.get_device_id():
            a = src.astype(
                dtype, order=order, copy=copy, casting=None, subok=None)
        else:
            a = src.copy(order=order).astype(
                dtype, copy=False, order=None, casting=None, subok=None)

        ndim = a._shape.size()
        if ndmin > ndim:
            if a is obj:
                # When `copy` is False, `a` is same as `obj`.
                a = a.view()
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a

    if hasattr(obj, '__cuda_array_interface__'):
        return array(core._convert_object_with_cuda_array_interface(obj),
                     dtype, copy, order, subok, ndmin)

    # obj is sequence, numpy array, scalar or the other type of object
    shape, elem_type, elem_dtype = _get_concat_shape(obj)
    if shape is not None and shape[-1] != 0:
        # obj is a non-empty sequence of ndarrays which share same shape
        # and dtype

        # resulting array is C order unless 'F' is explicitly specified
        # (i.e., it ignores order of element arrays in the sequence)
        order = (
            'F'
            if order is not None and len(order) >= 1 and order[0] in 'Ff'
            else 'C')
        ndim = len(shape)
        if ndmin > ndim:
            shape = (1,) * (ndmin - ndim) + shape

        if dtype is None:
            dtype = elem_dtype
        # Note: dtype might not be numpy.dtype in this place

        if issubclass(elem_type, numpy.ndarray):
            # obj is Seq[numpy.ndarray]
            return _send_numpy_array_list_to_gpu(
                obj, elem_dtype, dtype, shape, order, ndmin)

        # obj is Seq[cupy.ndarray]
        assert issubclass(elem_type, ndarray), elem_type
        lst = _flatten_list(obj)
        if len(shape) == 1:
            # convert each scalar (0-dim) ndarray to 1-dim
            lst = [cupy.expand_dims(x, 0) for x in lst]

        a =_manipulation.concatenate_method(lst, 0)
        a = a.reshape(shape)
        a = a.astype(dtype, order=order, copy=False, casting=None, subok=None)
        return a

    # obj is:
    # - numpy array
    # - scalar or sequence of scalar
    # - empty sequence or sequence with elements whose shapes or
    #   dtypes are unmatched
    # - other types

    # fallback to numpy array and send it to GPU
    # Note: dtype might not be numpy.dtype in this place
    return _send_object_to_gpu(obj, dtype, order, ndmin)


cdef tuple _get_concat_shape(object obj):
    # Returns a tuple of the following:
    # 1. concatenated shape if it can be converted to a single CuPy array by
    #    just concatenating it (i.e., the object is a (nested) sequence only
    #    which contains NumPy/CuPy array(s) with same shape and dtype).
    #    Returns None otherwise.
    # 2. type of the first item in the object
    # 3. dtype if the object is an array
    if isinstance(obj, (list, tuple)):
        return _get_concat_shape_impl(obj)
    return (None, None, None)


cdef tuple _get_concat_shape_impl(object obj):
    cdef obj_type = type(obj)
    if issubclass(obj_type, (numpy.ndarray, ndarray)):
        # obj.shape is () when obj.ndim == 0
        return (obj.shape, obj_type, obj.dtype)
    if isinstance(obj, (list, tuple)):
        shape = None
        typ = None
        dtype = None
        for elem in obj:
            # Find the head recursively if obj is a nested built-in list
            elem_shape, elem_type, elem_dtype = _get_concat_shape_impl(elem)

            # Use shape of the first element as the common shape.
            if shape is None:
                shape = elem_shape
                typ = elem_type
                dtype = elem_dtype

            # `elem` is not concatable or the shape and dtype does not match
            # with siblings.
            if (elem_shape is None
                    or shape != elem_shape
                    or dtype != elem_dtype):
                return (None, obj_type, None)

        if shape is None:
            shape = ()
        return (
            (len(obj),) + shape,
            typ,
            dtype)
    # scalar or object
    return (None, obj_type, None)


cdef list _flatten_list(object obj):
    ret = []
    if isinstance(obj, (list, tuple)):
        for elem in obj:
            ret += _flatten_list(elem)
        return ret
    return [obj]


cdef ndarray _send_object_to_gpu(obj, dtype, order, Py_ssize_t ndmin):
    if order is not None and len(order) >= 1 and order[0] in 'KAka':
        if isinstance(obj, numpy.ndarray) and obj.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
    a_cpu = numpy.array(obj, dtype=dtype, copy=False, order=order,
                        ndmin=ndmin)
    a_dtype = a_cpu.dtype  # converted to numpy.dtype
    if a_dtype.char not in '?bhilqBHILQefdFD':
        raise ValueError('Unsupported dtype %s' % a_dtype)
    cdef vector.vector[Py_ssize_t] a_shape = a_cpu.shape
    cdef ndarray a = ndarray(a_shape, dtype=a_dtype, order=order)
    if a_cpu.ndim == 0:
        a.fill(a_cpu)
        return a
    cdef Py_ssize_t nbytes = a.nbytes

    stream = stream_module.get_current_stream()
    cdef pinned_memory.PinnedMemoryPointer mem = (
        _alloc_async_transfer_buffer(nbytes))
    if mem is not None:
        src_cpu = numpy.frombuffer(mem, a_dtype, a_cpu.size)
        src_cpu[:] = a_cpu.ravel(order)
        a.data.copy_from_host_async(ctypes.c_void_p(mem.ptr), nbytes)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        a.data.copy_from_host(
            ctypes.c_void_p(a_cpu.__array_interface__['data'][0]),
            nbytes)

    return a


cdef ndarray _send_numpy_array_list_to_gpu(
        list arrays, src_dtype, dst_dtype,
        const vector.vector[Py_ssize_t]& shape,
        order, Py_ssize_t ndmin):

    a_dtype = _dtype.get_dtype(dst_dtype)  # convert to numpy.dtype
    if a_dtype.char not in '?bhilqBHILQefdFD':
        raise ValueError('Unsupported dtype %s' % a_dtype)
    cdef ndarray a  # allocate it after pinned memory is secured
    cdef size_t itemcount = internal.prod(shape)
    cdef size_t nbytes = itemcount * a_dtype.itemsize

    stream = stream_module.get_current_stream()
    cdef pinned_memory.PinnedMemoryPointer mem = (
        _alloc_async_transfer_buffer(nbytes))
    cdef size_t offset, length
    if mem is not None:
        # write concatenated arrays to the pinned memory directly
        src_cpu = (
            numpy.frombuffer(mem, a_dtype, itemcount)
            .reshape(shape, order=order))
        _concatenate_numpy_array(
            [numpy.expand_dims(e, 0) for e in arrays],
            0,
            _dtype.get_dtype(src_dtype),
            a_dtype,
            src_cpu)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host_async(ctypes.c_void_p(mem.ptr), nbytes)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        # fallback to numpy array and send it to GPU
        # Note: a_cpu.ndim is always >= 1
        a_cpu = numpy.array(arrays, dtype=a_dtype, copy=False, order=order,
                            ndmin=ndmin)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host(
            ctypes.c_void_p(a_cpu.__array_interface__['data'][0]),
            nbytes)

    return a


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
    except runtime.CUDARuntimeError as e:
        if e.status != runtime.cudaErrorMemoryAllocation:
            raise
        warnings.warn(
            'Using synchronous transfer as pinned memory ({} bytes) '
            'could not be allocated. '
            'This generally occurs because of insufficient host memory. '
            'The original error was: {}'.format(nbytes, e),
            util.PerformanceWarning)

    return None


cpdef ndarray internal_ascontiguousarray(ndarray a):
    if a._c_contiguous:
        return a
    newarray = core._ndarray_init(a._shape, a.dtype)
    _ufuncs.elementwise_copy(a, newarray)
    return newarray


cpdef ndarray internal_asfortranarray(ndarray a):
    cdef ndarray newarray
    cdef int m, n

    if a._f_contiguous:
        return a

    newarray = ndarray(a.shape, a.dtype, order='F')
    if (a._c_contiguous and a._shape.size() == 2 and
            (a.dtype == numpy.float32 or a.dtype == numpy.float64)):
        m, n = a.shape
        handle = device.get_cublas_handle()
        if a.dtype == numpy.float32:
            cublas.sgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                newarray.data.ptr, m)
        elif a.dtype == numpy.float64:
            cublas.dgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                newarray.data.ptr, m)
    else:
        _ufuncs.elementwise_copy(a, newarray)
    return newarray


cpdef ndarray ascontiguousarray(ndarray a, dtype=None):
    cdef bint same_dtype = False
    zero_dim = a._shape.size() == 0
    if dtype is None:
        same_dtype = True
        dtype = a.dtype
    else:
        dtype = _dtype.get_dtype(dtype)
        same_dtype = dtype == a.dtype

    if same_dtype and a._c_contiguous:
        if zero_dim:
            return _manipulation._ndarray_ravel(a, 'C')
        return a

    shape = (1,) if zero_dim else a.shape
    newarray = ndarray(shape, dtype)
    _ufuncs.elementwise_copy(a, newarray)
    return newarray


cpdef ndarray asfortranarray(ndarray a, dtype=None):
    cdef ndarray newarray
    cdef int m, n
    cdef bint same_dtype = False
    zero_dim = a._shape.size() == 0

    if dtype is None:
        dtype = a.dtype
        same_dtype = True
    else:
        dtype = _dtype.get_dtype(dtype)
        same_dtype = dtype == a.dtype

    if same_dtype and a._f_contiguous:
        if zero_dim:
            return _manipulation._ndarray_ravel(a, 'F')
        return a

    if same_dtype and not zero_dim:
        return internal_asfortranarray(a)

    newarray = ndarray((1,) if zero_dim else a.shape, dtype, order='F')
    _ufuncs.elementwise_copy(a, newarray)
    return newarray

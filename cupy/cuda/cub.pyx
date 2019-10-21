# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

import numpy

from cupy.core.core cimport ndarray, _internal_ascontiguousarray
from cupy.cuda cimport stream
from cupy.cuda.driver cimport Stream as Stream_t

cimport cython


###############################################################################
# Const
###############################################################################

cdef enum:
    CUPY_CUB_INT8 = 0
    CUPY_CUB_UINT8 = 1
    CUPY_CUB_INT16 = 2
    CUPY_CUB_UINT16 = 3
    CUPY_CUB_INT32 = 4
    CUPY_CUB_UINT32 = 5
    CUPY_CUB_INT64 = 6
    CUPY_CUB_UINT64 = 7
    CUPY_CUB_FLOAT16 = 8
    CUPY_CUB_FLOAT32 = 9
    CUPY_CUB_FLOAT64 = 10
    CUPY_CUB_COMPLEX64 = 11
    CUPY_CUB_COMPLEX128 = 12

###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cub.h':
    void cub_device_reduce(void*, size_t&, void*, void*, int, Stream_t,
                           int, int)
    void cub_device_segmented_reduce(void*, size_t&, void*, void*, int, void*,
                                     void*, Stream_t, int, int)
    size_t cub_device_reduce_get_workspace_size(void*, void*, int, Stream_t,
                                                int, int)
    size_t cub_device_segmented_reduce_get_workspace_size(
        void*, void*, int, void*, void*, Stream_t, int, int)

###############################################################################
# Python interface
###############################################################################


def _preprocess_array(ndarray arr, axis, bint keepdims):
    # if import at the top level, a segfault would happen when import cupy!
    from cupy.core._kernel import _get_axis
    from cupy.core._routines_manipulation import _transpose

    cdef tuple reduce_axis, out_axis, axis_permutes, out_shape
    cdef ndarray new_arr
    cdef Py_ssize_t contiguous_size = 1

    reduce_axis, out_axis = _get_axis(axis, arr._shape.size())
    axis_permutes = out_axis + reduce_axis
    if axis_permutes != tuple(range(len(arr.shape))):
        new_arr = _transpose(arr, axis_permutes)
    else:
        new_arr = arr
    new_arr = _internal_ascontiguousarray(new_arr)

    for axis in reduce_axis:
        contiguous_size *= arr.shape[axis]
    if not keepdims:
        out_shape = tuple([arr.shape[axis] for axis in out_axis])
    else:
        temp = []
        for axis in range(arr.ndim):
            if axis in out_axis:
                temp.append(arr.shape[axis])
            else:  # in reduce_axis
                temp.append(1)
        out_shape = tuple(temp)

    return new_arr, out_shape, contiguous_size


def device_reduce(ndarray x, int op, out=None, bint keepdims=False):
    cdef ndarray y
    cdef ndarray ws
    cdef int dtype_id, ndim_out
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    cdef Stream_t s

    ndim_out = keepdims
    if out is not None and out.ndim != ndim_out:
        raise ValueError(
            "output parameter for reduction operation sum has the wrong "
            "number of dimensions")
    if op < 0 or op > 2:
        raise ValueError("only CUPY_CUB_SUM, CUPY_CUB_MIN, and CUPY_CUB_MAX "
                         "are supported.")
    x = _internal_ascontiguousarray(x)
    y = ndarray((), x.dtype)
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    dtype_id = _get_dtype_id(x.dtype)
    s = <Stream_t>stream.get_current_stream_ptr()

    ws_size = cub_device_reduce_get_workspace_size(x_ptr, y_ptr, x.size, s,
                                                   op, dtype_id)
    ws = ndarray(ws_size, numpy.int8)
    ws_ptr = <void *>ws.data.ptr
    cub_device_reduce(ws_ptr, ws_size, x_ptr, y_ptr, x.size, s, op, dtype_id)

    if keepdims:
        y = y.reshape((1,))
    if out is not None:
        out[...] = y
        y = out
    return y


def device_segmented_reduce(ndarray x, int op, axis, out=None,
                            bint keepdims=False):
    # if import at the top level, a segfault would happen when import cupy!
    from cupy.creation.ranges import arange

    cdef ndarray x_reshaped, y, ws, offset
    cdef int dtype_id, n_segments
    cdef size_t ws_size
    cdef Py_ssize_t contiguous_size
    cdef tuple out_shape
    cdef void *x_ptr, *y_ptr, *ws_ptr, *offset_start_ptr
    cdef Stream_t s

    if op < 0 or op > 2:
        raise ValueError("only CUPY_CUB_SUM, CUPY_CUB_MIN, and CUPY_CUB_MAX "
                         "are supported.")

    # prepare input
    x_reshaped, out_shape, contiguous_size = _preprocess_array(x, axis,
                                                               keepdims)
    x_ptr = <void*>x_reshaped.data.ptr
    y = ndarray(out_shape, dtype=x_reshaped.dtype)
    y_ptr = <void*>y.data.ptr
    if out is not None and out.shape != out_shape:
        raise ValueError(
            "output parameter for reduction operation has the wrong shape")
    n_segments = x_reshaped.size//contiguous_size
    # CUB internally use int for offset...
    offset = arange(0, x_reshaped.size+1, contiguous_size, dtype=numpy.int32)
    offset_start_ptr = <void*>offset.data.ptr
    offset_end_ptr = <void*>((<int*><void*>offset.data.ptr)+1)
    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = _get_dtype_id(x_reshaped.dtype)

    # get workspace size and then fire up
    ws_size = cub_device_segmented_reduce_get_workspace_size(
        x_ptr, y_ptr, n_segments, offset_start_ptr, offset_end_ptr, s,
#        x_ptr, y_ptr, n_segments, offset_start_ptr, offset_start_ptr, s,
        op, dtype_id)
    ws = ndarray(ws_size, numpy.int8)
    ws_ptr = <void*>ws.data.ptr
    cub_device_segmented_reduce(ws_ptr, ws_size, x_ptr, y_ptr, n_segments,
                                offset_start_ptr, offset_end_ptr, s,
#                                offset_start_ptr, offset_start_ptr, s,
                                op, dtype_id)
    print("cub_device_segmented_reduce finished!")

    if out is not None:
        out[...] = y
        y = out
    return y


cpdef bint _cub_axis_compatible(axis, Py_ssize_t ndim):
    if ((axis is None) or ndim == 1 or axis == tuple(range(ndim))):
        return True
    return False


def can_use_device_reduce(x_dtype, Py_ssize_t ndim, int op, dtype=None,
                          axis=None):
    if not can_use_device_segmented_reduce(x_dtype, op, dtype):
        return False
    return _cub_axis_compatible(axis, ndim)


def can_use_device_segmented_reduce(x_dtype, int op, dtype=None):
    if dtype is None:
        if op == CUPY_CUB_SUM:
            # auto dtype:
            # CUB reduce_sum does not support dtype promotion.
            # See _sum_auto_dtype in cupy/core/_routines_math.pyx for which
            # dtypes are promoted.
            support_dtype = [numpy.int64, numpy.uint64,
                             numpy.float32, numpy.float64,
                             numpy.complex64, numpy.complex128]
        else:
            support_dtype = [numpy.int8, numpy.uint8,
                             numpy.int16, numpy.uint16,
                             numpy.int32, numpy.uint32,
                             numpy.int64, numpy.uint64,
                             numpy.float32, numpy.float64,
                             numpy.complex64, numpy.complex128]
    elif dtype == x_dtype:
        support_dtype = [numpy.int8, numpy.uint8,
                         numpy.int16, numpy.uint16,
                         numpy.int32, numpy.uint32,
                         numpy.int64, numpy.uint64,
                         numpy.float32, numpy.float64,
                         numpy.complex64, numpy.complex128]
    else:
        return False
    if x_dtype not in support_dtype:
        return False
    return True


def _get_dtype_id(dtype):
    if dtype == numpy.int8:
        ret = CUPY_CUB_INT8
    elif dtype == numpy.uint8:
        ret = CUPY_CUB_UINT8
    elif dtype == numpy.int16:
        ret = CUPY_CUB_INT16
    elif dtype == numpy.uint16:
        ret = CUPY_CUB_UINT16
    elif dtype == numpy.int32:
        ret = CUPY_CUB_INT32
    elif dtype == numpy.uint32:
        ret = CUPY_CUB_UINT32
    elif dtype == numpy.int64:
        ret = CUPY_CUB_INT64
    elif dtype == numpy.uint64:
        ret = CUPY_CUB_UINT64
    elif dtype == numpy.float32:
        ret = CUPY_CUB_FLOAT32
    elif dtype == numpy.float64:
        ret = CUPY_CUB_FLOAT64
    elif dtype == numpy.complex64:
        ret = CUPY_CUB_COMPLEX64
    elif dtype == numpy.complex128:
        ret = CUPY_CUB_COMPLEX128
    else:
        raise ValueError('Unsupported dtype ({})'.format(dtype))
    return ret

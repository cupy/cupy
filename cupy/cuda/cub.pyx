# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

import numpy

from cupy.core cimport core
from cupy.cuda cimport common

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
    void cub_reduce_sum(void*, void*, int, void*, size_t&, int)
    void cub_reduce_min(void*, void*, int, void*, size_t&, int)
    void cub_reduce_max(void*, void*, int, void*, size_t&, int)
    size_t cub_reduce_sum_get_workspace_size(void*, void*, int, int)
    size_t cub_reduce_min_get_workspace_size(void*, void*, int, int)
    size_t cub_reduce_max_get_workspace_size(void*, void*, int, int)

###############################################################################
# Python interface
###############################################################################


def reduce_sum(core.ndarray x, out=None, bint keepdims=False):
    cdef core.ndarray y
    cdef core.ndarray ws
    cdef int dtype_id, ndim_out
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    ndim_out = keepdims
    if out is not None and out.ndim != ndim_out:
        raise ValueError(
            "output parameter for reduction operation sum has the wrong "
            "number of dimensions")
    x = core.ascontiguousarray(x)
    y = core.ndarray((), x.dtype)
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    dtype_id = _get_dtype_id(x.dtype)
    ws_size = cub_reduce_sum_get_workspace_size(x_ptr, y_ptr, x.size, dtype_id)
    ws = core.ndarray(ws_size, numpy.int8)
    ws_ptr = <void *>ws.data.ptr
    cub_reduce_sum(x_ptr, y_ptr, x.size, ws_ptr, ws_size, dtype_id)
    if keepdims:
        y = y.reshape((1,))
    if out is not None:
        out[...] = y
        y = out
    return y


cpdef bint _cub_axis_compatible(axis, Py_ssize_t ndim):
    if ((axis is None) or ndim == 1 or axis == tuple(range(ndim))):
        return True
    return False


def can_use_reduce_sum(x_dtype, Py_ssize_t ndim, dtype=None, axis=None):
    if dtype is None:
        # auto dtype:
        # CUB reduce_sum does not support dtype promotion.
        # See _sum_auto_dtype in cupy/core/_routines_math.pyx for which dtypes
        # are promoted.
        support_dtype = [numpy.int64, numpy.uint64,
                         numpy.float32, numpy.float64,
                         numpy.complex64, numpy.complex128]
    elif dtype == x_dtype:
        support_dtype = [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
                         numpy.int32, numpy.uint32, numpy.int64, numpy.uint64,
                         numpy.float32, numpy.float64,
                         numpy.complex64, numpy.complex128]
    else:
        return False
    if x_dtype not in support_dtype:
        return False
    return _cub_axis_compatible(axis, ndim)


def reduce_min(core.ndarray x, out=None, bint keepdims=False):
    cdef core.ndarray y
    cdef core.ndarray ws
    cdef int dtype_id, ndim_out
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    ndim_out = keepdims
    if out is not None and out.ndim != ndim_out:
        raise ValueError(
            "output parameter for reduction operation sum has the wrong "
            "number of dimensions")
    x = core.ascontiguousarray(x)
    y = core.ndarray((), x.dtype)
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    dtype_id = _get_dtype_id(x.dtype)
    ws_size = cub_reduce_min_get_workspace_size(x_ptr, y_ptr, x.size, dtype_id)
    ws = core.ndarray(ws_size, numpy.int8)
    ws_ptr = <void *>ws.data.ptr
    cub_reduce_min(x_ptr, y_ptr, x.size, ws_ptr, ws_size, dtype_id)
    if keepdims:
        y = y.reshape((1,))
    if out is not None:
        out[...] = y
        y = out
    return y


def can_use_reduce_min(x_dtype, Py_ssize_t ndim, dtype=None, axis=None):
    if dtype is None or dtype == x_dtype:
        support_dtype = [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
                         numpy.int32, numpy.uint32, numpy.int64, numpy.uint64,
                         numpy.float32, numpy.float64,
                         numpy.complex64, numpy.complex128]
    else:
        return False
    if x_dtype not in support_dtype:
        return False
    return _cub_axis_compatible(axis, ndim)


def reduce_max(core.ndarray x, out=None, bint keepdims=False):
    cdef core.ndarray y
    cdef core.ndarray ws
    cdef int dtype_id, ndim_out
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    ndim_out = keepdims
    if out is not None and out.ndim != ndim_out:
        raise ValueError(
            "output parameter for reduction operation sum has the wrong "
            "number of dimensions")
    x = core.ascontiguousarray(x)
    y = core.ndarray((), x.dtype)
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    dtype_id = _get_dtype_id(x.dtype)
    ws_size = cub_reduce_max_get_workspace_size(x_ptr, y_ptr, x.size, dtype_id)
    ws = core.ndarray(ws_size, numpy.int8)
    ws_ptr = <void *>ws.data.ptr
    cub_reduce_max(x_ptr, y_ptr, x.size, ws_ptr, ws_size, dtype_id)
    if keepdims:
        y = y.reshape((1,))
    if out is not None:
        out[...] = y
        y = out
    return y


def can_use_reduce_max(x_dtype, Py_ssize_t ndim, dtype=None, axis=None):
    if dtype is None or dtype == x_dtype:
        support_dtype = [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16,
                         numpy.int32, numpy.uint32, numpy.int64, numpy.uint64,
                         numpy.float32, numpy.float64,
                         numpy.complex64, numpy.complex128]
    else:
        return False
    if x_dtype not in support_dtype:
        return False
    return _cub_axis_compatible(axis, ndim)


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

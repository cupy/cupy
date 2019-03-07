# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

import numpy

from cupy.core cimport core
from cupy.cuda cimport common

cimport cython


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cub.h' namespace 'cupy::cub':
    void _reduce_sum[T](void *, void *, int, void *, size_t)
    void _reduce_min[T](void *, void *, int, void *, size_t)
    void _reduce_max[T](void *, void *, int, void *, size_t)
    size_t _reduce_sum_get_workspace_size[T](void *, void *, int)
    size_t _reduce_min_get_workspace_size[T](void *, void *, int)
    size_t _reduce_max_get_workspace_size[T](void *, void *, int)


###############################################################################
# Python interface
###############################################################################

def reduce_sum(core.ndarray x, out=None):
    cdef core.ndarray y
    cdef core.ndarray ws
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    x = core.ascontiguousarray(x)
    if out is None:
        y = core.ndarray((), x.dtype)
    else:
        y = out
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    if x.dtype == numpy.int8:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_byte](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_byte](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint8:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_ubyte](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_ubyte](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int16:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_short](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_short](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint16:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_ushort](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_ushort](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int32:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_int](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_int](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint32:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_uint](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_uint](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int64:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_long](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_long](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint64:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_ulong](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_ulong](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.float32:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_float](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_float](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.float64:
        ws_size = _reduce_sum_get_workspace_size[common.cpy_double](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_sum[common.cpy_double](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    else:
        raise TypeError('Unsupported dtype: {}'.format(x.dtype))
    return y


def can_use_reduce_sum(dtype):
    ret = True
    if dtype is not None:
        support_dtype = [numpy.int8, numpy.uint8, numpy.int16, numpy.unit16,
                         numpy.int32, numpy.unit32, numpy.int64, numpy.unit64,
                         numpy.float32, numpy.float64]
        if dtype not in support_dtype:
            ret = False
    return ret


def reduce_min(core.ndarray x, out=None):
    cdef core.ndarray y
    cdef core.ndarray ws
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    x = core.ascontiguousarray(x)
    if out is None:
        y = core.ndarray((), x.dtype)
    else:
        y = out
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    if x.dtype == numpy.int8:
        ws_size = _reduce_min_get_workspace_size[common.cpy_byte](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_byte](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint8:
        ws_size = _reduce_min_get_workspace_size[common.cpy_ubyte](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_ubyte](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int16:
        ws_size = _reduce_min_get_workspace_size[common.cpy_short](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_short](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint16:
        ws_size = _reduce_min_get_workspace_size[common.cpy_ushort](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_ushort](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int32:
        ws_size = _reduce_min_get_workspace_size[common.cpy_int](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_int](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint32:
        ws_size = _reduce_min_get_workspace_size[common.cpy_uint](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_uint](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int64:
        ws_size = _reduce_min_get_workspace_size[common.cpy_long](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_long](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint64:
        ws_size = _reduce_min_get_workspace_size[common.cpy_ulong](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_ulong](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.float32:
        ws_size = _reduce_min_get_workspace_size[common.cpy_float](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_float](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.float64:
        ws_size = _reduce_min_get_workspace_size[common.cpy_double](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_min[common.cpy_double](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    else:
        raise TypeError('Unsupported dtype: {}'.format(x.dtype))
    return y


def can_use_reduce_min(dtype):
    ret = True
    if dtype is not None:
        support_dtype = [numpy.int8, numpy.uint8, numpy.int16, numpy.unit16,
                         numpy.int32, numpy.unit32, numpy.int64, numpy.unit64,
                         numpy.float32, numpy.float64]
        if dtype not in support_dtype:
            ret = False
    return ret


def reduce_max(core.ndarray x, out=None):
    cdef core.ndarray y
    cdef core.ndarray ws
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    x = core.ascontiguousarray(x)
    if out is None:
        y = core.ndarray((), x.dtype)
    else:
        y = out
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    if x.dtype == numpy.int8:
        ws_size = _reduce_max_get_workspace_size[common.cpy_byte](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_byte](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint8:
        ws_size = _reduce_max_get_workspace_size[common.cpy_ubyte](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_ubyte](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int16:
        ws_size = _reduce_max_get_workspace_size[common.cpy_short](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_short](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint16:
        ws_size = _reduce_max_get_workspace_size[common.cpy_ushort](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_ushort](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int32:
        ws_size = _reduce_max_get_workspace_size[common.cpy_int](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_int](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint32:
        ws_size = _reduce_max_get_workspace_size[common.cpy_uint](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_uint](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.int64:
        ws_size = _reduce_max_get_workspace_size[common.cpy_long](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_long](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.uint64:
        ws_size = _reduce_max_get_workspace_size[common.cpy_ulong](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_ulong](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.float32:
        ws_size = _reduce_max_get_workspace_size[common.cpy_float](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_float](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    elif x.dtype == numpy.float64:
        ws_size = _reduce_max_get_workspace_size[common.cpy_double](
            x_ptr, y_ptr, x.size)
        ws = core.ndarray(ws_size, numpy.int8)
        ws_ptr = <void *>ws.data.ptr
        _reduce_max[common.cpy_double](x_ptr, y_ptr, x.size, ws_ptr, ws_size)
    else:
        raise TypeError('Unsupported dtype: {}'.format(x.dtype))
    return y


def can_use_reduce_max(dtype):
    ret = True
    if dtype is not None:
        support_dtype = [numpy.int8, numpy.uint8, numpy.int16, numpy.unit16,
                         numpy.int32, numpy.unit32, numpy.int64, numpy.unit64,
                         numpy.float32, numpy.float64]
        if dtype not in support_dtype:
            ret = False
    return ret

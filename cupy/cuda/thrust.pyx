# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

import numpy

from cupy.cuda cimport common


###############################################################################
# Extern
###############################################################################

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void _sort[T](void *start, ptrdiff_t num)
    void _argsort[T](size_t *idx_start, void *data_start, size_t num)


###############################################################################
# Python interface
###############################################################################

cpdef sort(dtype, size_t start, size_t num):
    cdef void* ptr
    cdef Py_ssize_t n

    ptr = <void *>start
    n = <Py_ssize_t> num

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _sort[common.cpy_byte](ptr, n)
    elif dtype == numpy.uint8:
        _sort[common.cpy_ubyte](ptr, n)
    elif dtype == numpy.int16:
        _sort[common.cpy_short](ptr, n)
    elif dtype == numpy.uint16:
        _sort[common.cpy_ushort](ptr, n)
    elif dtype == numpy.int32:
        _sort[common.cpy_int](ptr, n)
    elif dtype == numpy.uint32:
        _sort[common.cpy_uint](ptr, n)
    elif dtype == numpy.int64:
        _sort[common.cpy_long](ptr, n)
    elif dtype == numpy.uint64:
        _sort[common.cpy_ulong](ptr, n)
    elif dtype == numpy.float32:
        _sort[common.cpy_float](ptr, n)
    elif dtype == numpy.float64:
        _sort[common.cpy_double](ptr, n)
    else:
        msg = "Sorting arrays with dtype '{}' is not supported"
        raise TypeError(msg.format(dtype))


cpdef argsort(dtype, size_t idx_start, size_t data_start, size_t num):
    cdef size_t *idx_ptr
    cdef void *data_ptr
    cdef size_t n

    idx_ptr = <size_t *>idx_start
    data_ptr = <void *>data_start
    n = <size_t>num

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _argsort[common.cpy_byte](idx_ptr, data_ptr, n)
    elif dtype == numpy.uint8:
        _argsort[common.cpy_ubyte](idx_ptr, data_ptr, n)
    elif dtype == numpy.int16:
        _argsort[common.cpy_short](idx_ptr, data_ptr, n)
    elif dtype == numpy.uint16:
        _argsort[common.cpy_ushort](idx_ptr, data_ptr, n)
    elif dtype == numpy.int32:
        _argsort[common.cpy_int](idx_ptr, data_ptr, n)
    elif dtype == numpy.uint32:
        _argsort[common.cpy_uint](idx_ptr, data_ptr, n)
    elif dtype == numpy.int64:
        _argsort[common.cpy_long](idx_ptr, data_ptr, n)
    elif dtype == numpy.uint64:
        _argsort[common.cpy_ulong](idx_ptr, data_ptr, n)
    elif dtype == numpy.float32:
        _argsort[common.cpy_float](idx_ptr, data_ptr, n)
    elif dtype == numpy.float64:
        _argsort[common.cpy_double](idx_ptr, data_ptr, n)
    else:
        msg = "Sorting arrays with dtype '{}' is not supported"
        raise TypeError(msg.format(dtype))

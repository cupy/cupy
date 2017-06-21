# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

cimport cython
import numpy

from cupy.cuda cimport common


###############################################################################
# Extern
###############################################################################

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void _sort[T](void *start, size_t ndim, size_t *shape)
    void _lexsort[T](size_t *idx_start, void *keys_start, size_t k, size_t n)
    void _argsort[T](size_t *idx_start, void *data_start, size_t num)


###############################################################################
# Python interface
###############################################################################

cpdef sort(dtype, size_t start, size_t ndim, size_t shape):
    cdef void *_start
    cdef size_t _ndim
    cdef size_t *_shape

    _start = <void *>start
    _ndim = <size_t>ndim
    _shape = <size_t *>shape

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _sort[common.cpy_byte](_start, _ndim, _shape)
    elif dtype == numpy.uint8:
        _sort[common.cpy_ubyte](_start, _ndim, _shape)
    elif dtype == numpy.int16:
        _sort[common.cpy_short](_start, _ndim, _shape)
    elif dtype == numpy.uint16:
        _sort[common.cpy_ushort](_start, _ndim, _shape)
    elif dtype == numpy.int32:
        _sort[common.cpy_int](_start, _ndim, _shape)
    elif dtype == numpy.uint32:
        _sort[common.cpy_uint](_start, _ndim, _shape)
    elif dtype == numpy.int64:
        _sort[common.cpy_long](_start, _ndim, _shape)
    elif dtype == numpy.uint64:
        _sort[common.cpy_ulong](_start, _ndim, _shape)
    elif dtype == numpy.float32:
        _sort[common.cpy_float](_start, _ndim, _shape)
    elif dtype == numpy.float64:
        _sort[common.cpy_double](_start, _ndim, _shape)
    else:
        msg = "Sorting arrays with dtype '{}' is not supported"
        raise TypeError(msg.format(dtype))


cpdef lexsort(dtype, size_t idx_start, size_t keys_start, size_t k, size_t n):

    idx_ptr = <size_t *>idx_start
    keys_ptr = <void *>keys_start

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _lexsort[common.cpy_byte](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.uint8:
        _lexsort[common.cpy_ubyte](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.int16:
        _lexsort[common.cpy_short](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.uint16:
        _lexsort[common.cpy_ushort](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.int32:
        _lexsort[common.cpy_int](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.uint32:
        _lexsort[common.cpy_uint](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.int64:
        _lexsort[common.cpy_long](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.uint64:
        _lexsort[common.cpy_ulong](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.float32:
        _lexsort[common.cpy_float](idx_ptr, keys_ptr, k, n)
    elif dtype == numpy.float64:
        _lexsort[common.cpy_double](idx_ptr, keys_ptr, k, n)
    else:
        raise TypeError('Sorting keys with dtype \'{}\' is not '
                        'supported'.format(dtype))


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

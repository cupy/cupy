# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

cimport cython
import numpy
from libcpp.vector cimport vector

from cupy.cuda cimport common


###############################################################################
# Extern
###############################################################################

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void _sort[T](void *, const vector.vector[ptrdiff_t]&)
    void _lexsort[T](size_t *, void *, size_t, size_t)
    void _argsort[T](size_t *, void *, void *, void *,
                     const vector.vector[ptrdiff_t]&)


###############################################################################
# Python interface
###############################################################################

cpdef sort(dtype, size_t start, vector.vector[ptrdiff_t]& shape):

    cdef void *_start
    _start = <void *>start

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _sort[common.cpy_byte](_start, shape)
    elif dtype == numpy.uint8:
        _sort[common.cpy_ubyte](_start, shape)
    elif dtype == numpy.int16:
        _sort[common.cpy_short](_start, shape)
    elif dtype == numpy.uint16:
        _sort[common.cpy_ushort](_start, shape)
    elif dtype == numpy.int32:
        _sort[common.cpy_int](_start, shape)
    elif dtype == numpy.uint32:
        _sort[common.cpy_uint](_start, shape)
    elif dtype == numpy.int64:
        _sort[common.cpy_long](_start, shape)
    elif dtype == numpy.uint64:
        _sort[common.cpy_ulong](_start, shape)
    elif dtype == numpy.float32:
        _sort[common.cpy_float](_start, shape)
    elif dtype == numpy.float64:
        _sort[common.cpy_double](_start, shape)
    else:
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is not '
                                  'supported'.format(dtype))


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


cpdef argsort(dtype, size_t idx_start, size_t data_start, size_t keys_start,
              size_t buff_start, vector.vector[ptrdiff_t]& shape):
    cdef size_t *_idx_start, *_keys_start
    cdef void *_data_start, *_buff_start

    _idx_start = <size_t *>idx_start
    _data_start = <void *>data_start
    _keys_start = <size_t *>keys_start
    _buff_start = <void *>buff_start

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _argsort[common.cpy_byte](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.uint8:
        _argsort[common.cpy_ubyte](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.int16:
        _argsort[common.cpy_short](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.uint16:
        _argsort[common.cpy_ushort](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.int32:
        _argsort[common.cpy_int](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.uint32:
        _argsort[common.cpy_uint](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.int64:
        _argsort[common.cpy_long](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.uint64:
        _argsort[common.cpy_ulong](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.float32:
        _argsort[common.cpy_float](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    elif dtype == numpy.float64:
        _argsort[common.cpy_double](
            _idx_start, _data_start, _keys_start, _buff_start, shape)
    else:
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is not '
                                  'supported'.format(dtype))

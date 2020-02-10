# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

import numpy

cimport cython  # NOQA
from libcpp cimport vector

from cupy.cuda cimport common
from cupy.cuda cimport memory
from cupy.cuda cimport stream


###############################################################################
# Memory Management
###############################################################################

cdef class _MemoryManager:
    cdef:
        dict memory

    def __init__(self):
        self.memory = dict()


cdef public char* cupy_malloc(void *m, ptrdiff_t size) with gil:
    if size == 0:
        return <char *>0
    cdef _MemoryManager mm = <_MemoryManager>m
    mem = memory.alloc(size)
    mm.memory[mem.ptr] = mem
    return <char *>mem.ptr


cdef public void cupy_free(void *m, char* ptr) with gil:
    if ptr == <char *>0:
        return
    cdef _MemoryManager mm = <_MemoryManager>m
    del mm.memory[<size_t>ptr]


###############################################################################
# Extern
###############################################################################

cdef extern from '../cuda/cupy_thrust.h' namespace 'cupy::thrust':
    void _sort[T](void *, size_t *, const vector.vector[ptrdiff_t]&, size_t,
                  void *)
    void _lexsort[T](size_t *, void *, size_t, size_t, size_t, void *)
    void _argsort[T](size_t *, void *, void *, const vector.vector[ptrdiff_t]&,
                     size_t, void *)

cdef extern from 'cupy_cuComplex.h':
    ctypedef struct cpy_complex64 'cuComplex':
        float x, y

    ctypedef struct cpy_complex128 'cuDoubleComplex':
        double x, y


###############################################################################
# Python interface
###############################################################################

cpdef sort(dtype, size_t data_start, size_t keys_start,
           const vector.vector[ptrdiff_t]& shape) except +:

    cdef void *_data_start
    cdef size_t *_keys_start
    cdef size_t _strm

    _data_start = <void *>data_start
    _keys_start = <size_t *>keys_start
    _strm = <size_t>(stream.get_current_stream_ptr())
    mem_obj = _MemoryManager()
    cdef void* mem = <void *>mem_obj

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _sort[common.cpy_byte](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint8:
        _sort[common.cpy_ubyte](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.int16:
        _sort[common.cpy_short](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint16:
        _sort[common.cpy_ushort](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.int32:
        _sort[common.cpy_int](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint32:
        _sort[common.cpy_uint](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.int64:
        _sort[common.cpy_long](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint64:
        _sort[common.cpy_ulong](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.float32:
        _sort[common.cpy_float](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.float64:
        _sort[common.cpy_double](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.complex64:
        _sort[cpy_complex64](_data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.complex128:
        _sort[cpy_complex128](_data_start, _keys_start, shape, _strm, mem)
    else:
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is not '
                                  'supported'.format(dtype))


cpdef lexsort(dtype, size_t idx_start, size_t keys_start,
              size_t k, size_t n) except +:

    cdef size_t _strm

    idx_ptr = <size_t *>idx_start
    keys_ptr = <void *>keys_start
    _strm = <size_t>(stream.get_current_stream_ptr())
    mem_obj = _MemoryManager()
    cdef void* mem = <void *>mem_obj

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _lexsort[common.cpy_byte](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.uint8:
        _lexsort[common.cpy_ubyte](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.int16:
        _lexsort[common.cpy_short](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.uint16:
        _lexsort[common.cpy_ushort](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.int32:
        _lexsort[common.cpy_int](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.uint32:
        _lexsort[common.cpy_uint](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.int64:
        _lexsort[common.cpy_long](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.uint64:
        _lexsort[common.cpy_ulong](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.float32:
        _lexsort[common.cpy_float](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.float64:
        _lexsort[common.cpy_double](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.complex64:
        _lexsort[cpy_complex64](idx_ptr, keys_ptr, k, n, _strm, mem)
    elif dtype == numpy.complex128:
        _lexsort[cpy_complex128](idx_ptr, keys_ptr, k, n, _strm, mem)
    else:
        raise TypeError('Sorting keys with dtype \'{}\' is not '
                        'supported'.format(dtype))


cpdef argsort(dtype, size_t idx_start, size_t data_start, size_t keys_start,
              const vector.vector[ptrdiff_t]& shape) except +:
    cdef size_t *_idx_start
    cdef size_t *_keys_start
    cdef void *_data_start
    cdef size_t _strm

    _idx_start = <size_t *>idx_start
    _data_start = <void *>data_start
    _keys_start = <size_t *>keys_start
    _strm = <size_t>(stream.get_current_stream_ptr())
    mem_obj = _MemoryManager()
    cdef void* mem = <void *>mem_obj

    # TODO(takagi): Support float16 and bool
    if dtype == numpy.int8:
        _argsort[common.cpy_byte](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint8:
        _argsort[common.cpy_ubyte](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.int16:
        _argsort[common.cpy_short](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint16:
        _argsort[common.cpy_ushort](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.int32:
        _argsort[common.cpy_int](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint32:
        _argsort[common.cpy_uint](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.int64:
        _argsort[common.cpy_long](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.uint64:
        _argsort[common.cpy_ulong](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.float32:
        _argsort[common.cpy_float](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.float64:
        _argsort[common.cpy_double](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.complex64:
        _argsort[cpy_complex64](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    elif dtype == numpy.complex128:
        _argsort[cpy_complex128](
            _idx_start, _data_start, _keys_start, shape, _strm, mem)
    else:
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is not '
                                  'supported'.format(dtype))

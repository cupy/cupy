# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

import numpy

cimport cython  # NOQA
from libc.stdint cimport intptr_t
from libcpp cimport vector

from cupy.cuda cimport common
from cupy.cuda cimport memory
from cupy.cuda cimport stream
from cupy_backends.cuda.api cimport runtime


###############################################################################
# Memory Management
###############################################################################

# Before attempting to refactor this part, read the discussion in #3212 first.

cdef class _MemoryManager:
    cdef:
        dict memory

    def __init__(self):
        self.memory = dict()


cdef public char* cupy_malloc(void *m, size_t size) with gil:
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

cdef extern from 'cupy_thrust.h':
    void thrust_sort(int, void *, size_t *, const vector.vector[ptrdiff_t]&,
                     intptr_t, void *)
    void thrust_lexsort(
        int, size_t *, void *, size_t, size_t, intptr_t, void *)
    void thrust_argsort(int, size_t *, void *, void *,
                        const vector.vector[ptrdiff_t]&, intptr_t, void *)

    # Build-time version
    int THRUST_VERSION


###############################################################################
# Python interface
###############################################################################


available = True


def get_build_version():
    return THRUST_VERSION


cpdef sort(dtype, intptr_t data_start, intptr_t keys_start,
           const vector.vector[ptrdiff_t]& shape) except +:
    cdef void* _data_start = <void*>data_start
    cdef size_t* _keys_start = <size_t*>keys_start
    cdef intptr_t _strm = stream.get_current_stream_ptr()
    cdef _MemoryManager mem_obj = _MemoryManager()
    cdef void* mem = <void*>mem_obj

    cdef int dtype_id
    try:
        dtype_id = common._get_dtype_id(dtype)
    except ValueError:
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is not '
                                  'supported'.format(dtype))
    if dtype_id == 8 and not common._is_fp16_supported():
        raise RuntimeError('either the GPU or the CUDA Toolkit does not '
                           'support fp16')

    thrust_sort(dtype_id, _data_start, _keys_start, shape, _strm, mem)


cpdef lexsort(dtype, intptr_t idx_start, intptr_t keys_start,
              size_t k, size_t n) except +:
    cdef size_t* idx_ptr = <size_t*>idx_start
    cdef void* keys_ptr = <void*>keys_start
    cdef intptr_t _strm = stream.get_current_stream_ptr()
    cdef _MemoryManager mem_obj = _MemoryManager()
    cdef void* mem = <void*>mem_obj

    cdef int dtype_id
    try:
        dtype_id = common._get_dtype_id(dtype)
    except ValueError:
        raise TypeError('Sorting keys with dtype \'{}\' is not '
                        'supported'.format(dtype))
    if dtype_id == 8 and not common._is_fp16_supported():
        raise RuntimeError('either the GPU or the CUDA Toolkit does not '
                           'support fp16')

    thrust_lexsort(dtype_id, idx_ptr, keys_ptr, k, n, _strm, mem)


cpdef argsort(dtype, intptr_t idx_start, intptr_t data_start,
              intptr_t keys_start,
              const vector.vector[ptrdiff_t]& shape) except +:
    cdef size_t*_idx_start = <size_t*>idx_start
    cdef void* _data_start = <void*>data_start
    cdef size_t* _keys_start = <size_t*>keys_start
    cdef intptr_t _strm = stream.get_current_stream_ptr()
    cdef _MemoryManager mem_obj = _MemoryManager()
    cdef void* mem = <void *>mem_obj

    cdef int dtype_id
    try:
        dtype_id = common._get_dtype_id(dtype)
    except ValueError:
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is not '
                                  'supported'.format(dtype))
    if dtype_id == 8 and not common._is_fp16_supported():
        raise RuntimeError('either the GPU or the CUDA Toolkit does not '
                           'support fp16')

    thrust_argsort(
        dtype_id, _idx_start, _data_start, _keys_start, shape, _strm, mem)

# distutils: language = c++

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t

import sys

import cupy


cdef extern from '../../../cupy/cuda/cupy_memory.h':
    cdef struct cupy_allocator_handle_t
    ctypedef cupy_allocator_handle_t cp_pool 'cupy_allocator_handle'
    cp_pool* get_cupy_allocator_handle()
    void destroy_cupy_allocator_handle(cp_pool*)
    void* cupy_malloc(cp_pool*, size_t)
    void cupy_free(cp_pool*, void*)


cdef _test_externel_access_to_cupy_pool() except +:
    cdef cp_pool* handle = get_cupy_allocator_handle()
    cdef size_t size = 100*sizeof(int)
    cdef int* h_ptr = <int*>PyMem_Malloc(size)
    cdef int i

    # initialize an array on host
    for i in range(100):
        h_ptr[i] = i + 3

    # allocate an array on device
    mod = sys.modules[__name__]
    cdef void* d_ptr = cupy_malloc(handle, size)
    mem = cupy.cuda.memory.UnownedMemory(<intptr_t>d_ptr, size, mod)
    memptr = cupy.cuda.memory.MemoryPointer(mem, 0)

    # transfer host data to device, and wrap it with cupy.ndarray
    cupy.cuda.runtime.memcpy(<intptr_t>d_ptr, <intptr_t>h_ptr, size,
                             cupy.cuda.runtime.memcpyHostToDevice)
    d_arr = cupy.ndarray((100,), dtype=cupy.int32, memptr=memptr)

    # check the integrity of the device data
    assert (d_arr == 3 + cupy.arange(100, dtype=cupy.int32)).all()

    # deallocation
    del d_arr
    cupy_free(handle, d_ptr)
    destroy_cupy_allocator_handle(handle)
    PyMem_Free(<void*>h_ptr)


def test_externel_access_to_cupy_pool():
    _test_externel_access_to_cupy_pool()

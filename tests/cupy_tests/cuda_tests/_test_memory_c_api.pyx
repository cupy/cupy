# distutils: language = c++

from libc.stdint cimport intptr_t
from libcpp cimport vector
from cupy.core.core cimport ndarray

import sys

import cupy


cdef extern from '../../../cupy/cuda/cupy_memory.h':
    cdef cppclass cupy_device_allocator:
        cupy_device_allocator() except +
        void* malloc(size_t)
        void free(void*)


cdef _test_externel_access_to_cupy_pool() except +:
    cdef cupy_device_allocator alloc
    cdef size_t size = 100*sizeof(int)
    cdef vector.vector[int] h_arr
    cdef int i

    # initialize an array on host
    h_arr.resize(100)
    for i in range(100):
        h_arr[i] = 2 * i
    cdef int* h_ptr = h_arr.data()

    # allocate an array on device
    mod = sys.modules[__name__]
    cdef void* d_ptr = alloc.malloc(size)
    mem = cupy.cuda.memory.UnownedMemory(<intptr_t>d_ptr, 0, None)
    memptr = cupy.cuda.memory.MemoryPointer(mem, 0)

    # transfer host data to device, and wrap it with cupy.ndarray
    cupy.cuda.runtime.memcpy(<intptr_t>d_ptr, <intptr_t>h_ptr, size, cupy.cuda.runtime.memcpyHostToDevice)

    d_arr = cupy.ndarray((100,), dtype=cupy.int32, memptr=memptr)

    # check the integrity of the device data
    assert (d_arr == 2 * cupy.arange(100, dtype=cupy.int32)).all()

    # deallocation
    del d_arr
    alloc.free(d_ptr)


def test_externel_access_to_cupy_pool():
    _test_externel_access_to_cupy_pool()

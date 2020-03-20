cimport cython  # NOQA

from libc.stdint cimport intptr_t
from libcpp cimport vector
from libcpp cimport map

from cupy.cuda cimport device


@cython.no_gc
cdef class BaseMemory:

    cdef:
        public intptr_t ptr
        public size_t size
        public int device_id


@cython.final
cdef class MemoryPointer:

    cdef:
        readonly intptr_t ptr
        readonly int device_id
        readonly BaseMemory mem

    cdef _init(self, BaseMemory mem, ptrdiff_t offset)

    cpdef copy_from_device(self, MemoryPointer src, size_t size)
    cpdef copy_from_device_async(self, MemoryPointer src, size_t size,
                                 stream=?)
    cpdef copy_from_host(self, mem, size_t size)
    cpdef copy_from_host_async(self, mem, size_t size, stream=?)
    cpdef copy_from(self, mem, size_t size)
    cpdef copy_from_async(self, mem, size_t size, stream=?)
    cpdef copy_to_host(self, mem, size_t size)
    cpdef copy_to_host_async(self, mem, size_t size, stream=?)
    cpdef memset(self, int value, size_t size)
    cpdef memset_async(self, int value, size_t size, stream=?)

    @staticmethod
    cdef _set_peer_access(int device, int peer)


cpdef MemoryPointer alloc(size)


cpdef set_allocator(allocator=*)
cpdef get_allocator()


cdef class MemoryPool:

    cdef:
        object _pools

    cpdef MemoryPointer malloc(self, size_t size)
    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef size_t n_free_blocks(self)
    cpdef size_t used_bytes(self)
    cpdef size_t free_bytes(self)
    cpdef size_t total_bytes(self)
    cpdef set_limit(self, size=?, fraction=?)
    cpdef size_t get_limit(self)


@cython.no_gc
cdef class CFunctionAllocatorMemory(BaseMemory):

    cdef:
        intptr_t _param
        intptr_t _free_func


cdef class CFunctionAllocator:

    cdef:
        intptr_t _param
        intptr_t _malloc_func
        intptr_t _free_func
        object _owner

    cpdef MemoryPointer malloc(self, size_t size)


###############################################################################
# Expose CuPy memory pool to C/C++ (migrated from cupy/cuda/thrust.pyx)       #
#                                                                             #
# We recommend using "cupy_device_allocator" from cupy/cuda/cupy_memory.h     #
# instead of using the two below functions directly.                          #
###############################################################################

cdef api void* cupy_c_malloc(size_t size) with gil
cdef api void cupy_c_free(void* ptr) with gil

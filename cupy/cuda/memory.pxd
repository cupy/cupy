cimport cython  # NOQA

from libcpp cimport vector
from libcpp cimport map

from cupy.cuda cimport device


@cython.no_gc
cdef class BaseMemory:

    cdef:
        public size_t ptr
        public Py_ssize_t size
        public int device_id


@cython.final
cdef class MemoryPointer:

    cdef:
        readonly size_t ptr
        readonly int device_id
        readonly BaseMemory mem

    cdef _init(self, BaseMemory mem, Py_ssize_t offset)

    cpdef copy_from_device(self, MemoryPointer src, Py_ssize_t size)
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


cdef class Allocator:

    cpdef MemoryPointer malloc(self, Py_ssize_t size)


cdef class MallocAllocator(Allocator):

    cdef:
        object _malloc


cdef class BaseMemoryPool(Allocator):

    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)


cdef class BaseMultiDeviceMemoryPool(BaseMemoryPool):

    cdef:
        object _pools

    cpdef BaseMemoryPool create_single_device_memory_pool(self)


# Default memory pool.
cdef class MemoryPool(BaseMultiDeviceMemoryPool):

    cdef:
        object _allocator


# External memory pool that may define malloc, free, etc. outside CuPy.
cdef class ExternalMemoryPool(BaseMultiDeviceMemoryPool):

    cdef:
        object _single_device_memory_pool_args


cpdef set_allocator(allocator=*)


cpdef MemoryPointer alloc(size)

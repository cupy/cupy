cimport cython  # NOQA

from libcpp cimport vector
from libcpp cimport map

from cupy.cuda cimport device


@cython.no_gc
cdef class Memory:

    cdef:
        public size_t ptr
        public Py_ssize_t size
        public int device_id


@cython.final
cdef class MemoryPointer:

    cdef:
        readonly size_t ptr
        readonly int device_id
        readonly Memory mem

    cdef _init(self, Memory mem, Py_ssize_t offset)

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


cpdef MemoryPointer alloc(size)


cpdef set_allocator(allocator=*)


cdef class MemoryPool:

    cdef:
        object _pools

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)

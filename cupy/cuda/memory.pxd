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


cdef class MemoryPointer:

    cdef:
        readonly size_t ptr
        readonly int device_id
        readonly Memory mem

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


cpdef MemoryPointer alloc(Py_ssize_t size)


cpdef set_allocator(allocator=*)


cdef class SingleDeviceMemoryPool:

    cdef:
        object _allocator

        # Map from memory pointer of the chunk (size_t) to the corresponding
        # Chunk object. All chunks currently allocated to the application from
        # this pool are stored.
        # `_in_use_lock` must be acquired to access.
        dict _in_use

        # Map from stream pointer (int) to its arena (list) for the stream.
        # `_free_lock` must be acquired to access.
        dict _free

        object __weakref__
        object _weakref
        object _free_lock
        object _in_use_lock
        readonly Py_ssize_t _allocation_unit_size
        readonly int _device_id

        # Map from stream pointer to its arena index.
        # `_free_lock` must be acquired to access.
        map.map[size_t, vector.vector[int]] _index

    cpdef MemoryPointer _alloc(self, Py_ssize_t size)
    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef MemoryPointer _malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)
    cpdef Py_ssize_t _round_size(self, Py_ssize_t size)
    cpdef int _bin_index_from_size(self, Py_ssize_t size)
    cpdef list _arena(self, size_t stream_ptr)
    cdef vector.vector[int]* _arena_index(self, size_t stream_ptr)
    cpdef _append_to_free_list(self, Py_ssize_t size, chunk, size_t stream_ptr)
    cpdef bint _remove_from_free_list(self, Py_ssize_t size,
                                      chunk, size_t stream_ptr) except *

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

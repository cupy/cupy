cimport cython  # NOQA

from libc.stdint cimport int8_t
from libcpp cimport vector
from libcpp cimport map

from cupy.cuda cimport device


@cython.no_gc
cdef class BaseMemory:

    cdef:
        public size_t ptr
        public Py_ssize_t size
        public int device_id


@cython.no_gc
cdef class Memory(BaseMemory):
    """Memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.
    """
    pass


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


cpdef MemoryPointer alloc(size)


cpdef set_allocator(allocator=*)


cdef class BaseSingleDeviceMemoryPool:

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)


cdef class SingleDeviceMemoryPool(BaseSingleDeviceMemoryPool):

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
        readonly int _device_id

        # Map from stream pointer to its arena index.
        # `_free_lock` must be acquired to access.
        map.map[size_t, vector.vector[int]] _index
        map.map[size_t, vector.vector[int8_t]] _flag

    cpdef list _arena(self, size_t stream_ptr)
    cdef inline vector.vector[int]* _arena_index(self, size_t stream_ptr)
    cdef vector.vector[int8_t]* _arena_flag(self, size_t stream_ptr)
    cdef Memory _try_malloc(self, Py_ssize_t size)
    cpdef MemoryPointer _alloc(self, Py_ssize_t rounded_size)
    cpdef MemoryPointer _malloc(self, Py_ssize_t size)

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)


#@cython.final
cdef class ExternalSingleDeviceMemoryPool(BaseSingleDeviceMemoryPool):

    cdef:
        object __weakref__
        object _weakref
        object _allocator
        object _free
        readonly int _device_id



cdef class BaseMemoryPool:

    cdef:
        object _pools
        object _allocator

    cpdef BaseSingleDeviceMemoryPool create_single_device_memory_pool(self)

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free_all_blocks(self, stream=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)


#@cython.final
cdef class MemoryPool(BaseMemoryPool):

    pass


# @cython.final
cdef class ExternalMemoryPool(BaseMemoryPool):

    cdef:
        object _single_device_memory_pool_args

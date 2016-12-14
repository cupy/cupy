from cupy.cuda cimport device as _device

cdef class Memory:

    cdef:
        public _device.Device device
        public size_t ptr
        public Py_ssize_t size

cdef class ManagedMemory(Memory):

    cpdef prefetch(self, stream)
    cpdef advise(self, int advice, _device.Device device)

cdef class Chunk:

    cdef:
        readonly device.Device device
        readonly object mem
        readonly size_t ptr
        readonly Py_ssize_t offset
        readonly Py_ssize_t size
        public Chunk prev
        public Chunk next
        public bint in_use

cdef class MemoryPointer:

    cdef:
        readonly _device.Device device
        readonly object mem
        readonly size_t ptr

    cpdef copy_from_device(self, MemoryPointer src, Py_ssize_t size)
    cpdef copy_from_device_async(self, MemoryPointer src, size_t size, stream)
    cpdef copy_from_host(self, mem, size_t size)
    cpdef copy_from_host_async(self, mem, size_t size, stream)
    cpdef copy_from(self, mem, size_t size)
    cpdef copy_from_async(self, mem, size_t size, stream)
    cpdef copy_to_host(self, mem, size_t size)
    cpdef copy_to_host_async(self, mem, size_t size, stream)
    cpdef memset(self, int value, size_t size)
    cpdef memset_async(self, int value, size_t size, stream)


cpdef MemoryPointer alloc(Py_ssize_t size)


cpdef set_allocator(allocator=*)


cdef class PooledMemory(Memory):

    cdef:
        object pool

    cpdef free(self)


cdef class SingleDeviceMemoryPool:

    cdef:
        object _alloc
        dict _in_use
        object _free
        object __weakref__
        object _weakref
        readonly Py_ssize_t _allocation_unit_size
        readonly Py_ssize_t _initial_bins_size

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)
    cpdef Py_ssize_t _round_size(self, Py_ssize_t size)
    cpdef Py_ssize_t _bin_index_from_size(self, Py_ssize_t size)
    cpdef void _grow_free_if_necessary(self, Py_ssize_t size)
    cpdef tuple _split(self, Chunk chunk, Py_ssize_t size)
    cpdef Chunk _merge(self, Chunk head, Chunk remaining)

cdef class MemoryPool:

    cdef:
        object _pools

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free_all_blocks(self)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)

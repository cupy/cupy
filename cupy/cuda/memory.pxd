from cupy.cuda cimport device

cdef class Memory:

    cdef:
        public device.Device device
        public size_t ptr
        public Py_ssize_t size


cdef class MemoryPointer:

    cdef:
        readonly device.Device device
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
        Py_ssize_t _allocation_unit_size

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)


cdef class MemoryPool:

    cdef:
        object _pools

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)

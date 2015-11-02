from cupy.cuda cimport device

cdef class Memory:

    cdef:
        public Py_ssize_t size
        public size_t ptr
        device.Device _device


cdef class MemoryPointer:

    cdef:
        public object mem
        public size_t ptr
        device.Device _device

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
        dict _in_use
        object _free, _alloc
        object __weakref__
        object _weakref

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_free(self)


cdef class MemoryPool:

    cdef:
        object _pools

    cpdef MemoryPointer malloc(self, Py_ssize_t size)

from cupy.cuda cimport memory

cdef class PinnedMemoryPointer:

    cdef:
        readonly object mem
        readonly size_t ptr
        Py_ssize_t _shape[1]
        Py_ssize_t _strides[1]

    cpdef Py_ssize_t size(self)
    cpdef copy_from_device(self, memory.MemoryPointer src, Py_ssize_t size)
    cpdef copy_from_device_async(self, memory.MemoryPointer src,
                                 Py_ssize_t size, stream)
    cpdef copy_to_device(self, memory.MemoryPointer dst, Py_ssize_t size)
    cpdef copy_to_device_async(self, memory.MemoryPointer dst,
                               Py_ssize_t size, stream)

cpdef _add_to_watch_list(event, obj)


cpdef PinnedMemoryPointer alloc_pinned_memory(Py_ssize_t size)


cpdef set_pinned_memory_allocator(allocator=*)


cdef class PinnedMemoryPool:

    cdef:
        object _alloc
        dict _in_use
        object _free
        object __weakref__
        object _weakref
        object _lock
        Py_ssize_t _allocation_unit_size

    cpdef PinnedMemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self)
    cpdef n_free_blocks(self)


cdef class PinnedMemory:

    cdef:
        public size_t ptr
        public Py_ssize_t size


cdef class PinnedMemoryPointer:

    cdef:
        readonly object mem
        readonly size_t ptr
        Py_ssize_t _shape[1]
        Py_ssize_t _strides[1]

    cpdef Py_ssize_t size(self)


cpdef PinnedMemoryPointer alloc_pinned_memory(Py_ssize_t size)


cpdef set_pinned_memory_allocator(allocator=*)


cdef class PooledPinnedMemory(PinnedMemory):

    cdef:
        object pool

    cpdef free(self)


cdef class PinnedMemoryPool:

    cdef:
        object _alloc
        dict _in_use
        object _free
        object __weakref__
        object _weakref
        Py_ssize_t _allocation_unit_size

    cpdef PinnedMemoryPointer malloc(self, Py_ssize_t size)
    cpdef free(self, size_t ptr, Py_ssize_t size)
    cpdef free_all_blocks(self)
    cpdef n_free_blocks(self)

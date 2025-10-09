from libc.stdint cimport intptr_t
from libcpp.mutex cimport recursive_mutex


cdef class PinnedMemoryPointer:

    cdef:
        readonly object mem
        readonly intptr_t ptr
        Py_ssize_t _shape[1]
        Py_ssize_t _strides[1]

    cpdef size_t size(self)


cpdef _add_to_watch_list(event, obj)


cpdef PinnedMemoryPointer alloc_pinned_memory(size_t size)


cpdef set_pinned_memory_allocator(allocator=*)


cdef class PinnedMemoryPool:

    cdef:
        object _alloc
        dict _in_use
        object _free
        object __weakref__
        object _weakref
        size_t _allocation_unit_size
        # NOTE: Never use `lock()` outside a nogil statement, because
        # almost anything could release the GIL and then deadlocks happen
        # if another thread tries to lock also (without the GIL released).
        # You can try with `try_lock()` first though.
        recursive_mutex _lock

    cpdef PinnedMemoryPointer malloc(self, size_t size)
    cpdef free(self, intptr_t ptr, size_t size)
    cpdef free_all_blocks(self)
    cpdef n_free_blocks(self)


cpdef bint is_memory_pinned(intptr_t data) except*

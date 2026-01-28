# distutils: language = c++

import collections
import weakref

from cupy_backends.cuda.api import runtime

from cupy._core cimport internal
from cupy_backends.cuda.api cimport runtime
from cupy import _util


class PinnedMemory(object):

    """Pinned memory allocation on host.

    This class provides a RAII interface of the pinned memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, size_t size, unsigned int flags=0):
        self.size = size
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.hostAlloc(size, flags)

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.ptr:
            runtime.freeHost(self.ptr)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr


cdef class PinnedMemoryPointer:

    """Pointer of a pinned memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (PinnedMemory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        ~PinnedMemoryPointer.mem (PinnedMemory): The device memory buffer.
        ~PinnedMemoryPointer.ptr (int): Pointer to the place within the buffer.
    """

    def __init__(self, mem, ptrdiff_t offset):
        self.mem = mem
        self.ptr = mem.ptr + offset

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr

    def __add__(x, y):
        """Adds an offset to the pointer."""
        cdef PinnedMemoryPointer self
        cdef ptrdiff_t offset
        if isinstance(x, PinnedMemoryPointer):
            self = x
            offset = <ptrdiff_t?>y
        else:
            self = <PinnedMemoryPointer?>y
            offset = <ptrdiff_t?>x
        return PinnedMemoryPointer(
            self.mem, self.ptr - self.mem.ptr + offset)

    def __iadd__(self, ptrdiff_t offset):
        """Adds an offset to the pointer in place."""
        self.ptr += offset
        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, ptrdiff_t offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    cpdef size_t size(self) noexcept:
        return self.mem.size - (self.ptr - self.mem.ptr)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        size = self.size()

        self._shape[0] = size
        self._strides[0] = 1

        buffer.buf = <void*>self.ptr
        buffer.format = 'b'
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = size
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cdef class _EventWatcher:
    cdef:
        list events
        # NOTE: Never use `lock()` outside a nogil statement, because
        # almost anything could release the GIL and then deadlocks happen
        # if another thread tries to lock also (without the GIL released).
        # You can try with `try_lock()` first though.
        recursive_mutex _lock

    def __init__(self):
        self.events = []

    cpdef add(self, event, obj):
        """ Add event to be monitored.

        The ``obj`` are automatically released when the event done.

        Args:
            event (cupy.cuda.Event): The CUDA event to be monitored.
            obj: The object to be held.
        """
        if not self._lock.try_lock():
            with nogil:
                self._lock.lock()
        try:
            self._check_and_release_without_lock()
            if event.done:
                return
            self.events.append((event, obj))
        finally:
            self._lock.unlock()

    cpdef check_and_release(self):
        """ Check and release completed events.

        """
        if not self.events:
            return

        if not self._lock.try_lock():
            with nogil:
                self._lock.lock()
        try:
            self._check_and_release_without_lock()
        finally:
            self._lock.unlock()

    cpdef _check_and_release_without_lock(self):
        while self.events and self.events[0][0].done:
            del self.events[0]


cpdef PinnedMemoryPointer _malloc(size_t size):
    mem = PinnedMemory(size, runtime.hostAllocPortable)
    return PinnedMemoryPointer(mem, 0)


cdef object _current_allocator = _malloc
cdef _EventWatcher _watcher = _EventWatcher()


cpdef _add_to_watch_list(event, obj):
    """ Add event to be monitored.

    The ``obj`` are automatically released when the event done.

    Args:
        event (cupy.cuda.Event): The CUDA event to be monitored.
        obj: The object to be held.
    """
    _watcher.add(event, obj)


cpdef PinnedMemoryPointer alloc_pinned_memory(size_t size):
    """Calls the current allocator.

    Use :func:`~cupy.cuda.set_pinned_memory_allocator` to change the current
    allocator.

    Args:
        size (int): Size of the memory allocation.

    Returns:
        ~cupy.cuda.PinnedMemoryPointer: Pointer to the allocated buffer.

    """
    _watcher.check_and_release()
    return _current_allocator(size)


cpdef set_pinned_memory_allocator(allocator=None):
    """Sets the current allocator for the pinned memory.

    Args:
        allocator (function): CuPy pinned memory allocator. It must have the
            same interface as the :func:`cupy.cuda.alloc_pinned_memory`
            function, which takes the buffer size as an argument and returns
            the device buffer of that size. When ``None`` is specified, raw
            memory allocator is used (i.e., memory pool is disabled).

    """
    global _current_allocator
    if allocator is None:
        allocator = _malloc
    _current_allocator = allocator


class PooledPinnedMemory(PinnedMemory):

    """Memory allocation for a memory pool.

    As the instance of this class is created by memory pool allocator, users
    should not instantiate it manually.

    """

    def __init__(self, mem, pool):
        self.ptr = mem.ptr
        self.size = mem.size
        self.pool = pool

    def free(self):
        """Releases the memory buffer and sends it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        pool = self.pool()
        if pool and self.ptr != 0:
            pool.free(self.ptr, self.size)
        self.ptr = 0
        self.size = 0

    __del__ = free


cdef class PinnedMemoryPool:

    """Memory pool for pinned memory on the host.

    Note that it preserves all allocated memory buffers even if the user
    explicitly release the one. Those released memory buffers are held by the
    memory pool as *free blocks*, and reused for further memory allocations of
    the same size.

    Args:
        allocator (function): The base CuPy pinned memory allocator. It is
            used for allocating new blocks when the blocks of the required
            size are all in use.

    """
    def __init__(self, allocator=_malloc):
        self._in_use = {}
        self._free = collections.defaultdict(list)
        self._alloc = allocator
        self._weakref = weakref.ref(self)
        self._allocation_unit_size = 512

    cpdef PinnedMemoryPointer malloc(self, size_t size):
        cdef list free
        cdef size_t unit

        if size == 0:
            return PinnedMemoryPointer(PinnedMemory(0), 0)

        # Round up the memory size to fit memory alignment of cudaHostAlloc
        unit = self._allocation_unit_size
        size = internal.clp2(((size + unit - 1) // unit) * unit)
        if not self._lock.try_lock():
            with nogil:
                self._lock.lock()
        try:
            free = self._free[size]
            if free:
                mem = free.pop()
            else:
                try:
                    mem = self._alloc(size).mem
                except runtime.CUDARuntimeError as e:
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    self.free_all_blocks()
                    mem = self._alloc(size).mem

            self._in_use[mem.ptr] = mem
        finally:
            self._lock.unlock()
        pmem = PooledPinnedMemory(mem, self._weakref)
        return PinnedMemoryPointer(pmem, 0)

    cpdef free(self, intptr_t ptr, size_t size):
        cdef list free
        if not self._lock.try_lock():
            with nogil:
                self._lock.lock()
        try:
            mem = self._in_use.pop(ptr, None)
            if mem is None:
                raise RuntimeError('Cannot free out-of-pool memory')
            free = self._free[size]
            free.append(mem)
        finally:
            self._lock.unlock()

    cpdef free_all_blocks(self):
        """Release free all blocks."""
        _watcher.check_and_release()
        if not self._lock.try_lock():
            with nogil:
                self._lock.lock()
        try:
            self._free.clear()
        finally:
            self._lock.unlock()

    cpdef n_free_blocks(self):
        """Count the total number of free blocks.

        Returns:
            int: The total number of free blocks.
        """
        cdef Py_ssize_t n = 0
        if not self._lock.try_lock():
            with nogil:
                self._lock.lock()
        try:
            for v in self._free.values():
                n += len(v)
        finally:
            self._lock.unlock()
        return n


cpdef bint is_memory_pinned(intptr_t data) except*:
    cdef runtime.PointerAttributes attrs = runtime.pointerGetAttributes(data)
    return (attrs.type == runtime.memoryTypeHost)

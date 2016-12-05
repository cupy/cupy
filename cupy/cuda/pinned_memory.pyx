# distutils: language = c++

import collections
import weakref

import six

from cupy.cuda import runtime

from cupy.cuda cimport runtime


cdef class PinnedMemory:

    """Pinned memory allocation on host.

    This class provides a RAII interface of the pinned memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size, unsigned int flags=0):
        self.size = size
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.hostAlloc(size, flags)

    def __dealloc__(self):
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
        mem (PinnedMemory): The device memory buffer.
        ptr (int): Pointer to the place within the buffer.
    """

    def __init__(self, PinnedMemory mem, Py_ssize_t offset):
        self.mem = mem
        self.ptr = mem.ptr + offset

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr

    def __add__(x, y):
        """Adds an offset to the pointer."""
        cdef PinnedMemoryPointer self
        cdef Py_ssize_t offset
        if isinstance(x, PinnedMemoryPointer):
            self = x
            offset = <Py_ssize_t?>y
        else:
            self = <PinnedMemoryPointer?>y
            offset = <Py_ssize_t?>x
        return PinnedMemoryPointer(
            self.mem, self.ptr - self.mem.ptr + offset)

    def __iadd__(self, Py_ssize_t offset):
        """Adds an offset to the pointer in place."""
        self.ptr += offset
        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, Py_ssize_t offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    cpdef Py_ssize_t size(self):
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

    def __getsegcount__(self, Py_ssize_t *lenp):
        if lenp != NULL:
            lenp[0] = self.size()
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0:
            raise SystemError("accessing non-existent buffer segment")
        p[0] = <void*>self.ptr
        return self.size()

    def __getwritebuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0:
            raise SystemError("accessing non-existent buffer segment")
        p[0] = <void*>self.ptr
        return self.size()


cpdef PinnedMemoryPointer _malloc(Py_ssize_t size):
    mem = PinnedMemory(size, runtime.hostAllocPortable)
    return PinnedMemoryPointer(mem, 0)


cdef object _current_allocator = _malloc


cpdef PinnedMemoryPointer alloc_pinned_memory(Py_ssize_t size):
    """Calls the current allocator.

    Use :func:`~cupy.cuda.set_pinned_memory_allocator` to change the current
    allocator.

    Args:
        size (int): Size of the memory allocation.

    Returns:
        ~cupy.cuda.PinnedMemoryPointer: Pointer to the allocated buffer.

    """
    return _current_allocator(size)


cpdef set_pinned_memory_allocator(allocator=_malloc):
    """Sets the current allocator.

    Args:
        allocator (function): CuPy pinned memory allocator. It must have the
            same interface as the :func:`cupy.cuda.alloc_alloc_pinned_memory`
            function, which takes the buffer size as an argument and returns
            the device buffer of that size.

    """
    global _current_allocator
    _current_allocator = allocator


cdef class PooledPinnedMemory(PinnedMemory):

    """Memory allocation for a memory pool.

    As the instance of this class is created by memory pool allocator, users
    should not instantiate it manually.

    """

    def __init__(self, PinnedMemory mem, pool):
        self.ptr = mem.ptr
        self.size = mem.size
        self.pool = pool

    def __dealloc__(self):
        if self.ptr != 0:
            self.free()

    cpdef free(self):
        """Releases the memory buffer and sends it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        pool = self.pool()
        if pool and self.ptr != 0:
            pool.free(self.ptr, self.size)
        self.ptr = 0
        self.size = 0


cdef class PinnedMemoryPool:

    """Memory pool on the host.

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

    cpdef PinnedMemoryPointer malloc(self, Py_ssize_t size):
        cdef list free
        cdef PinnedMemory mem

        if size == 0:
            return PinnedMemoryPointer(PinnedMemory(0), 0)

        # Round up the memory size to fit memory alignment of cudaHostAlloc
        unit = self._allocation_unit_size
        size = (((size + unit - 1) // unit) * unit)
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
        pmem = PooledPinnedMemory(mem, self._weakref)
        return PinnedMemoryPointer(pmem, 0)

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef list free
        cdef PinnedMemory mem
        mem = self._in_use.pop(ptr, None)
        if mem is None:
            raise RuntimeError('Cannot free out-of-pool memory')
        free = self._free[size]
        free.append(mem)

    cpdef free_all_blocks(self):
        """Release free all blocks."""
        self._free = collections.defaultdict(list)

    cpdef n_free_blocks(self):
        """Count the total number of free blocks.

        Returns:
            int: The total number of free blocks.
        """
        cdef Py_ssize_t n = 0
        for v in six.itervalues(self._free):
            n += len(v)
        return n

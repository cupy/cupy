# distutils: language = c++

import collections
import ctypes
import gc
import warnings
import weakref

import six

from cupy.cuda import runtime

from cupy.cuda cimport device
from cupy.cuda cimport runtime

_debug = False

cdef class Memory:

    """Memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        device (cupy.cuda.Device): Device whose memory the pointer refers to.
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device = None
        self.ptr = 0
        if size > 0:
            self.device = device.Device()
            self.ptr = runtime.malloc(size)

    def __dealloc__(self):
        if self.ptr:
            runtime.free(self.ptr)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr


cdef set _peer_access_checked = set()


cpdef _set_peer_access(int device, int peer):
    device_pair = device, peer

    if device_pair in _peer_access_checked:
        return
    cdef int can_access = runtime.deviceCanAccessPeer(device, peer)
    _peer_access_checked.add(device_pair)
    if not can_access:
        return

    cdef int current = runtime.getDevice()
    runtime.setDevice(device)
    try:
        runtime.deviceEnablePeerAccess(peer)
    finally:
        runtime.setDevice(current)


cdef class MemoryPointer:

    """Pointer to a point on a device memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (Memory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        device (cupy.cuda.Device): Device whose memory the pointer refers to.
        mem (Memory): The device memory buffer.
        ptr (int): Pointer to the place within the buffer.
    """

    def __init__(self, mem, Py_ssize_t offset):
        assert mem.ptr > 0 or offset == 0
        self.mem = mem
        self.device = mem.device
        self.ptr = mem.ptr + offset

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr

    def __add__(x, y):
        """Adds an offset to the pointer."""
        cdef MemoryPointer self
        cdef Py_ssize_t offset
        if isinstance(x, MemoryPointer):
            self = x
            offset = <Py_ssize_t?>y
        else:
            self = <MemoryPointer?>y
            offset = <Py_ssize_t?>x
        assert self.ptr > 0 or offset == 0
        return MemoryPointer(self.mem,
                             self.ptr - self.mem.ptr + offset)

    def __iadd__(self, Py_ssize_t offset):
        """Adds an offset to the pointer in place."""
        assert self.ptr > 0 or offset == 0
        self.ptr += offset
        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, Py_ssize_t offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    cpdef set_ptr(self, size_t ptr):
        """Sets ptr."""
        self.ptr = ptr

    cpdef copy_from_device(self, MemoryPointer src, Py_ssize_t size):
        """Copies a memory sequence from a (possibly different) device.

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            _set_peer_access(src.device.id, self.device.id)
            runtime.memcpy(self.ptr, src.ptr, size,
                           runtime.memcpyDefault)

    cpdef copy_from_device_async(self, MemoryPointer src, size_t size, stream):
        """Copies a memory from a (possibly different) device asynchronously.

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            _set_peer_access(src.device.id, self.device.id)
            runtime.memcpyAsync(self.ptr, src.ptr, size,
                                runtime.memcpyDefault, stream.ptr)

    cpdef copy_from_host(self, mem, size_t size):
        """Copies a memory sequence from the host memory.

        Args:
            mem (ctypes.c_void_p): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpy(self.ptr, mem.value, size,
                           runtime.memcpyHostToDevice)

    cpdef copy_from_host_async(self, mem, size_t size, stream):
        """Copies a memory sequence from the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Source memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            runtime.memcpyAsync(self.ptr, mem.value, size,
                                runtime.memcpyHostToDevice, stream.ptr)

    cpdef copy_from(self, mem, size_t size):
        """Copies a memory sequence from a (possibly different) device or host.

        This function is a useful interface that selects appropriate one from
        :meth:`~cupy.cuda.MemoryPointer.copy_from_device` and
        :meth:`~cupy.cuda.MemoryPointer.copy_from_host`.

        Args:
            mem (ctypes.c_void_p or cupy.cuda.MemoryPointer): Source memory
                pointer.
            size (int): Size of the sequence in bytes.

        """
        if isinstance(mem, MemoryPointer):
            self.copy_from_device(mem, size)
        else:
            self.copy_from_host(mem, size)

    cpdef copy_from_async(self, mem, size_t size, stream):
        """Copies a memory sequence from an arbitrary place asynchronously.

        This function is a useful interface that selects appropriate one from
        :meth:`~cupy.cuda.MemoryPointer.copy_from_device_async` and
        :meth:`~cupy.cuda.MemoryPointer.copy_from_host_async`.

        Args:
            mem (ctypes.c_void_p or cupy.cuda.MemoryPointer): Source memory
                pointer.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if isinstance(mem, MemoryPointer):
            self.copy_from_device_async(mem, size, stream)
        else:
            self.copy_from_host_async(mem, size, stream)

    cpdef copy_to_host(self, mem, size_t size):
        """Copies a memory sequence to the host memory.

        Args:
            mem (ctypes.c_void_p): Target memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpy(mem.value, self.ptr, size,
                           runtime.memcpyDeviceToHost)

    cpdef copy_to_host_async(self, mem, size_t size, stream):
        """Copies a memory sequence to the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Target memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            runtime.memcpyAsync(mem.value, self.ptr, size,
                                runtime.memcpyDeviceToHost, stream.ptr)

    cpdef memset(self, int value, size_t size):
        """Fills a memory sequence by constant byte value.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memset(self.ptr, value, size)

    cpdef memset_async(self, int value, size_t size, stream):
        """Fills a memory sequence by constant byte value asynchronously.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            runtime.memsetAsync(self.ptr, value, size, stream.ptr)


cpdef MemoryPointer _malloc(Py_ssize_t size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


cdef object _current_allocator = _malloc


cpdef MemoryPointer alloc(Py_ssize_t size):
    """Calls the current allocator.

    Use :func:`~cupy.cuda.set_allocator` to change the current allocator.

    Args:
        size (int): Size of the memory allocation.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

    """
    return _current_allocator(size)


cpdef set_allocator(allocator=_malloc):
    """Sets the current allocator.

    Args:
        allocator (function): CuPy memory allocator. It must have the same
            interface as the :func:`cupy.cuda.alloc` function, which takes the
            buffer size as an argument and returns the device buffer of that
            size.

    """
    global _current_allocator
    _current_allocator = allocator


cdef class PooledMemory(Memory):

    """Memory allocation for a memory pool.

    The instance of this class is created by memory pool allocator, so user
    should not instantiate it by hand.

    """

    def __init__(self, Memory mem, pool):
        self.ptr = mem.ptr
        self.size = mem.size
        self.device = mem.device
        self.pool = pool

    def __dealloc__(self):
        if self.ptr != 0:
            self.free()

    cpdef free(self):
        """Frees the memory buffer and returns it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        pool = self.pool()
        if pool and self.ptr != 0:
            pool.free(self.ptr, self.size)
        self.ptr = 0
        self.size = 0
        self.device = None


cdef class SingleDeviceMemoryPool:

    """Memory pool implementation for single device."""

    def __init__(self, allocator=_malloc):
        self._in_use = {}
        self._in_use_memptr = {}
        self._free = collections.defaultdict(list)
        self._alloc = allocator
        self._weakref = weakref.ref(self)
        # cudaMalloc() is aligned to at least 512 bytes
        # cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
        self._allocation_unit_size = 512

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        cdef list free
        cdef Memory mem

        if size == 0:
            return MemoryPointer(Memory(0), 0)

        # Round up the memory size to fit memory alignment of cudaMalloc
        unit = self._allocation_unit_size
        size = (((size + unit - 1) // unit) * unit)
        free = self._free[size]
        if free:
            mem = free.pop()
        else:
            try:
                mem = self._alloc(size).mem
            except runtime.CUDARuntimeError as e:
                runtime.deviceSynchronize()
                if e.status != runtime.errorMemoryAllocation:
                    raise
                if _debug:
                    print('# memory.pyx: free_all_blocks() is called.')
                self.free_all_blocks()
                try:
                    mem = self._alloc(size).mem
                except runtime.CUDARuntimeError as e:
                    runtime.deviceSynchronize()
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    if _debug:
                        print('# memory.pyx: total_bytes:{}, size:{}'
                              .format(self.total_bytes(), size))
                        print('# memory.pyx: realloc_all() is called.')
                    self.realloc_all()
                    gc.collect()
                    mem = self._alloc(size).mem

        pmem = PooledMemory(mem, self._weakref)
        memptr = MemoryPointer(pmem, 0)
        self._in_use[mem.ptr] = mem
        self._in_use_memptr[mem.ptr] = weakref.ref(memptr)
        return memptr

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef list free
        cdef Memory mem
        mem = self._in_use.pop(ptr, None)
        memptr = self._in_use_memptr.pop(ptr, None)
        if mem is None:
            raise RuntimeError('Cannot free out-of-pool memory')
        free = self._free[size]
        free.append(mem)

    cpdef free_all_blocks(self):
        self._free.clear()

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef size_t realloc(self, size_t ptr, Py_ssize_t new_size):
        """Reallocates the given area of memory."""
        cdef Memory cur_mem
        cdef Memory new_mem
        cdef MemoryPointer memptr
        cur_size = self._in_use[ptr].size
        try:
            new_mem = self._alloc(new_size).mem
        except runtime.CUDARuntimeError as e:
            if e.status != runtime.errorMemoryAllocation:
                raise
            runtime.deviceSynchronize()
            new_mem = None
        if new_mem is None:
            return ptr
        # memory copy
        cur_mem = self._in_use.pop(ptr, None)
        _size = min(cur_size, new_size)
        runtime.memcpy(new_mem.ptr, cur_mem.ptr, _size,
                       runtime.memcpyDeviceToDevice)
        runtime.deviceSynchronize()
        del cur_mem
        # update info
        memptr = self._in_use_memptr.pop(ptr, None)()
        memptr.mem.ptr = new_mem.ptr
        memptr.mem.size = new_mem.size
        memptr.set_ptr(new_mem.ptr)
        self._in_use[new_mem.ptr] = new_mem
        self._in_use_memptr[new_mem.ptr] = weakref.ref(memptr)

        return new_mem.ptr

    cpdef realloc_all(self):
        """Reallocates the memory of all in-use arrays"""
        _in_use = collections.defaultdict(list)
        for ptr in self._in_use.keys():
            mem = self._in_use[ptr]
            _in_use[mem.size].append(ptr)

        for size in sorted(_in_use.keys()):
            for ptr in sorted(_in_use[size]):
                new_ptr = self.realloc(ptr, size)
                if new_ptr > ptr:
                    self.realloc(new_ptr, size)

    cpdef n_free_blocks(self):
        cdef Py_ssize_t n = 0
        for v in six.itervalues(self._free):
            n += len(v)
        return n

    cpdef used_bytes(self):
        cdef Py_ssize_t size = 0
        for mem in six.itervalues(self._in_use):
            size += mem.size
        return size

    cpdef free_bytes(self):
        cdef Py_ssize_t size = 0
        for free_list in six.itervalues(self._free):
            for mem in free_list:
                size += mem.size
        return size

    cpdef total_bytes(self):
        return self.used_bytes() + self.free_bytes()


cdef class MemoryPool(object):

    """Memory pool for all devices on the machine.

    A memory pool preserves any allocations even if they are freed by the user.
    Freed memory buffers are held by the memory pool as *free blocks*, and they
    are reused for further memory allocations of the same sizes. The allocated
    blocks are managed for each device, so one instance of this class can be
    used for multiple devices.

    .. note::
       When the allocation is skipped by reusing the pre-allocated block, it
       does not call ``cudaMalloc`` and therefore CPU-GPU synchronization does
       not occur. It makes interleaves of memory allocations and kernel
       invocations very fast.

    .. note::
       The memory pool holds allocated blocks without freeing as much as
       possible. It makes the program hold most of the device memory, which may
       make other CUDA programs running in parallel out-of-memory situation.

    Args:
        allocator (function): The base CuPy memory allocator. It is used for
            allocating new blocks when the blocks of the required size are all
            in use.

    """

    def __init__(self, allocator=_malloc):
        self._pools = collections.defaultdict(
            lambda: SingleDeviceMemoryPool(allocator))

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        """Allocates the memory, from the pool if possible.

        This method can be used as a CuPy memory allocator. The simplest way to
        use a memory pool as the default allocator is the following code::

           set_allocator(MemoryPool().malloc)

        Args:
            size (int): Size of the memory buffer to allocate in bytes.

        Returns:
            ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.malloc(size)

    cpdef free_all_blocks(self):
        """Release free blocks."""
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        mp.free_all_blocks()

    cpdef free_all_free(self):
        """Release free blocks."""
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        """Count the total number of free blocks.

        Returns:
            int: The total number of free blocks.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.n_free_blocks()

    cpdef used_bytes(self):
        """Get the total number of bytes used.

        Returns:
            int: The total number of bytes used.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.used_bytes()

    cpdef free_bytes(self):
        """Get the total number of bytes acquired but not used in the pool.

        Returns:
            int: The total number of bytes acquired but not used in the pool.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.free_bytes()

    cpdef total_bytes(self):
        """Get the total number of bytes acquired in the pool.

        Returns:
            int: The total number of bytes acquired in the pool.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.total_bytes()

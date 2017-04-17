# distutils: language = c++

import collections
import ctypes
import gc
import warnings
import weakref

import six

from cupy.cuda import runtime
from cupy.cuda import stream

from cupy.cuda cimport device as device_mod
from cupy.cuda cimport runtime


_cuda_version = runtime.runtimeGetVersion()


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
            self.device = device_mod.Device()
            self.ptr = runtime.malloc(size)

    def __dealloc__(self):
        if self.ptr:
            runtime.free(self.ptr)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr


cdef class ManagedMemory(Memory):

    """Managed memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA managed memory
    allocation.

    Args:
        device (cupy.cuda.Device): Device whose memory the pointer refers to.
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device = None
        self.ptr = 0
        if size > 0:
            self.device = device_mod.Device()
            self.ptr = runtime.mallocManaged(size)

    cpdef prefetch(self, stream):
        """ Prefetch memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream.
        """
        runtime.memPrefetchAsync(self.ptr, self.size, self.device.id,
                                 stream.ptr)

    cpdef advise(self, int advise, device_mod.Device device):
        """ Advise about the usage of this memory.

        Args:
            advics (int): Advise to be applied for this memory.
            device (cupy.cuda.Device): Device to apply the advice for.

        """
        runtime.memAdvise(self.ptr, self.size, advise, device.id)


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

cdef class Chunk:

    """A chunk points to a device memory.

    A chunk might be a splitted memory block from a larger allocation.
    The prev/next pointers contruct a doubly-linked list of memory addresses
    sorted by base address that must be contiguous.

    Args:
        mem (Memory): The device memory buffer.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.

    Attributes:
        device (cupy.cuda.Device): Device whose memory the pointer refers to.
        mem (Memory): The device memory buffer.
        ptr (int): Memory address.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        prev (Chunk): prev memory pointer if split from a larger allocation
        next (Chunk): next memory pointer if split from a larger allocation
        in_use (boolen): in_use flag
    """

    def __init__(self, mem, Py_ssize_t offset, Py_ssize_t size):
        assert mem.ptr > 0 or offset == 0
        self.mem = mem
        self.device = mem.device
        self.ptr = mem.ptr + offset
        self.offset = offset
        self.size = size
        self.prev = None
        self.next = None
        self.in_use = False

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


cpdef MemoryPointer malloc_managed(Py_ssize_t size):
    """Allocate managed memory.

    Args:
        size (int): Size of the memory allocation in bytes.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.
    """
    mem = ManagedMemory(size)
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

    def __init__(self, Chunk chunk, pool):
        self.device = chunk.device
        self.ptr = chunk.ptr
        self.size = chunk.size
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
    """Memory pool implementation for single device.

    - The allocator attempts to find the smallest cached block that will fit
      the requested size. If the block is larger than the requested size,
      it may be split. If no block is found, the allocator will delegate to
      cudaMalloc.
    - If the cudaMalloc fails, the allocator will free all cached blocks that
      are not split and retry the allocation.
    """

    def __init__(self, allocator=_malloc):
        # cudaMalloc() is aligned to at least 512 bytes
        # cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
        self._allocation_unit_size = 512
        self._initial_bins_size = 1024
        self._in_use = {}
        self._free = [[] for i in range(self._initial_bins_size)]
        self._alloc = allocator
        self._weakref = weakref.ref(self)

    cpdef Py_ssize_t _round_size(self, Py_ssize_t size):
        """Round up the memory size to fit memory alignment of cudaMalloc."""
        unit = self._allocation_unit_size
        return (((size + unit - 1) // unit) * unit)

    cpdef Py_ssize_t _bin_index_from_size(self, Py_ssize_t size):
        """Get appropriate bins (_free) index from the memory size"""
        unit = self._allocation_unit_size
        return (size - 1) // unit

    cpdef void _grow_free_if_necessary(self, Py_ssize_t size):
        """Extend bins (_free) size if necessary"""
        current_size = len(self._free)
        if current_size >= size:
            return
        growth_size = size - current_size
        growth = [[] for i in range(growth_size)]
        self._free.extend(growth)

    cpdef tuple _split(self, Chunk chunk, Py_ssize_t size):
        """Split contiguous block of a larger allocation"""
        assert not chunk.in_use
        assert chunk.size >= size
        if chunk.size == size:
            return (chunk, None)
        cdef Chunk head
        cdef Chunk remaining
        cdef int index
        head = Chunk(chunk.mem, chunk.offset, size)
        remaining = Chunk(chunk.mem, chunk.offset + size, chunk.size - size)
        if chunk.prev is not None:
            head.prev = chunk.prev
            chunk.prev.next = head
        if chunk.next is not None:
            remaining.next = chunk.next
            chunk.next.prev = remaining
        head.next = remaining
        remaining.prev = head
        index = self._bin_index_from_size(remaining.size)
        self._free[index].append(remaining)
        return (head, remaining)

    cpdef Chunk _merge(self, Chunk head, Chunk remaining):
        """Merge previously splitted block (chunk)"""
        assert not head.in_use
        assert not remaining.in_use
        cdef Chunk merged
        size = head.size + remaining.size
        merged = Chunk(head.mem, head.offset, size)
        if head.prev is not None:
            merged.prev = head.prev
            merged.prev.next = merged
        if remaining.next is not None:
            merged.next = remaining.next
            merged.next.prev = merged
        return merged

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        cdef list free_list = None
        cdef Chunk chunk = None
        cdef MemoryPointer memptr
        cdef Memory mem

        if size == 0:
            return MemoryPointer(Memory(0), 0)

        size = self._round_size(size)
        index = self._bin_index_from_size(size)
        # find best-fit, or a smallest larger allocation
        length = len(self._free)
        for i in range(index, length):
            free_list = self._free[i]
            if free_list:
                chunk = free_list.pop()
                chunk, _remaining = self._split(chunk, size)
                break

        # cudaMalloc if not found
        if chunk is None:
            try:
                mem = self._alloc(size).mem
            except runtime.CUDARuntimeError as e:
                if e.status != runtime.errorMemoryAllocation:
                    raise
                self.free_all_blocks()
                try:
                    mem = self._alloc(size).mem
                except runtime.CUDARuntimeError as e:
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    gc.collect()
                    mem = self._alloc(size).mem
            chunk = Chunk(mem, 0, size)

        chunk.in_use = True
        self._in_use[chunk.ptr] = chunk
        pmem = PooledMemory(chunk, self._weakref)
        return MemoryPointer(pmem, 0)

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef Chunk chunk
        cdef int index

        chunk = self._in_use.pop(ptr, None)
        if chunk is None:
            raise RuntimeError('Cannot free out-of-pool memory')

        chunk.in_use = False
        if chunk.next and not chunk.next.in_use:
            index = self._bin_index_from_size(chunk.next.size)
            self._free[index].remove(chunk.next)
            chunk = self._merge(chunk, chunk.next)

        if chunk.prev and not chunk.prev.in_use:
            index = self._bin_index_from_size(chunk.prev.size)
            self._free[index].remove(chunk.prev)
            chunk = self._merge(chunk.prev, chunk)

        index = self._bin_index_from_size(chunk.size)
        self._grow_free_if_necessary(index + 1)
        self._free[index].append(chunk)

    cpdef free_all_blocks(self):
        # Free all **non-split** chunks
        cdef list free_list
        cdef Chunk chunk
        for free_list in self._free:
            for chunk in free_list:
                if not chunk.prev and not chunk.next:
                    free_list.remove(chunk)

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        cdef Py_ssize_t n = 0
        for v in self._free:
            n += len(v)
        return n

    cpdef used_bytes(self):
        cdef Py_ssize_t size = 0
        for chunk in self._in_use.itervalues():
            size += chunk.size
        return size

    cpdef free_bytes(self):
        cdef Py_ssize_t size = 0
        for free_list in self._free:
            for chunk in free_list:
                size += chunk.size
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

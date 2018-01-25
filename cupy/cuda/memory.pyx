# distutils: language = c++
import collections
import ctypes
import gc
import threading
import warnings
import weakref

from fastrlock cimport rlock
from libcpp cimport algorithm

from cupy.cuda import driver
from cupy.cuda import runtime

from cupy.cuda cimport device
from cupy.cuda cimport memory_hook
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module


thread_local = threading.local()


cdef inline _ensure_context(int device_id):

    """Ensure that CUcontext bound to the calling host thread exists.

    See discussion on https://github.com/cupy/cupy/issues/72 for details.
    """

    if not hasattr(thread_local, 'seen_devices'):
        thread_local.seen_devices = set()
    if device_id not in thread_local.seen_devices:
        if driver.ctxGetCurrent() == 0:
            # Call Runtime API to establish context on this host thread.
            runtime.memGetInfo()
        thread_local.seen_devices.add(device_id)


class OutOfMemoryError(MemoryError):

    def __init__(self, size, total):
        msg = 'out of memory to allocate %d bytes ' \
              '(total %d bytes)' % (size, total)
        super(OutOfMemoryError, self).__init__(msg)


class Memory(object):

    """Memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        ~Memory.device (cupy.cuda.Device): Device whose memory the pointer
            refers to.
        ~Memory.size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device = None
        self.ptr = 0
        if size > 0:
            self.device = device.Device()
            self.ptr = runtime.malloc(size)

    def __del__(self):
        if self.ptr:
            runtime.free(self.ptr)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr


class ManagedMemory(Memory):

    """Managed memory (Unified memory) allocation on a CUDA device.

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
            self.device = device.Device()
            self.ptr = runtime.mallocManaged(size)

    def prefetch(self, stream):
        """(experimental) Prefetch memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream.
        """
        runtime.memPrefetchAsync(self.ptr, self.size, self.device.id,
                                 stream.ptr)

    def advise(self, int advise, device.Device device):
        """(experimental) Advise about the usage of this memory.

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
        stream_ptr (size_t): Raw stream handle of cupy.cuda.Stream

    Attributes:
        device (cupy.cuda.Device): Device whose memory the pointer refers to.
        mem (Memory): The device memory buffer.
        ptr (size_t): Memory address.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        prev (Chunk): prev memory pointer if split from a larger allocation
        next (Chunk): next memory pointer if split from a larger allocation
        stream_ptr (size_t): Raw stream handle of cupy.cuda.Stream
    """

    def __init__(self, mem, Py_ssize_t offset, Py_ssize_t size, stream_ptr):
        assert mem.ptr > 0 or offset == 0
        self.mem = mem
        self.device = mem.device
        self.ptr = mem.ptr + offset
        self.offset = offset
        self.size = size
        self.stream_ptr = stream_ptr
        self.prev = None
        self.next = None


cdef class MemoryPointer:

    """Pointer to a point on a device memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (Memory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        ~MemoryPointer.device (cupy.cuda.Device): Device whose memory the
            pointer refers to.
        mem (Memory): The device memory buffer.
        ptr (size_t): Pointer to the place within the buffer.
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

    cpdef copy_from_device_async(self, MemoryPointer src, size_t size,
                                 stream=None):
        """Copies a memory from a (possibly different) device asynchronously.

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

        """
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        if size > 0:
            _set_peer_access(src.device.id, self.device.id)
            runtime.memcpyAsync(self.ptr, src.ptr, size,
                                runtime.memcpyDefault, stream_ptr)

    cpdef copy_from_host(self, mem, size_t size):
        """Copies a memory sequence from the host memory.

        Args:
            mem (ctypes.c_void_p): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpy(self.ptr, mem.value, size,
                           runtime.memcpyHostToDevice)

    cpdef copy_from_host_async(self, mem, size_t size, stream=None):
        """Copies a memory sequence from the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Source memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

        """
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        if size > 0:
            runtime.memcpyAsync(self.ptr, mem.value, size,
                                runtime.memcpyHostToDevice, stream_ptr)

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

    cpdef copy_from_async(self, mem, size_t size, stream=None):
        """Copies a memory sequence from an arbitrary place asynchronously.

        This function is a useful interface that selects appropriate one from
        :meth:`~cupy.cuda.MemoryPointer.copy_from_device_async` and
        :meth:`~cupy.cuda.MemoryPointer.copy_from_host_async`.

        Args:
            mem (ctypes.c_void_p or cupy.cuda.MemoryPointer): Source memory
                pointer.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

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

    cpdef copy_to_host_async(self, mem, size_t size, stream=None):
        """Copies a memory sequence to the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Target memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

        """
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        if size > 0:
            runtime.memcpyAsync(mem.value, self.ptr, size,
                                runtime.memcpyDeviceToHost, stream_ptr)

    cpdef memset(self, int value, size_t size):
        """Fills a memory sequence by constant byte value.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memset(self.ptr, value, size)

    cpdef memset_async(self, int value, size_t size, stream=None):
        """Fills a memory sequence by constant byte value asynchronously.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

        """
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        if size > 0:
            runtime.memsetAsync(self.ptr, value, size, stream_ptr)


cpdef MemoryPointer _malloc(Py_ssize_t size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_managed(Py_ssize_t size):
    """Allocate managed memory (unified memory).

    This method can be used as a CuPy memory allocator. The simplest way to
    use a managed memory as the default allocator is the following code::

        set_allocator(malloc_managed)

    The advantage using managed memory in CuPy is that device memory
    oversubscription is possible for GPUs that have a non-zero value for the
    device attribute cudaDevAttrConcurrentManagedAccess.
    CUDA >= 8.0 with GPUs later than or equal to Pascal is preferrable.

    Read more at: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#axzz4qygc1Ry1  # NOQA

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


cpdef set_allocator(allocator=None):
    """Sets the current allocator for GPU memory.

    Args:
        allocator (function): CuPy memory allocator. It must have the same
            interface as the :func:`cupy.cuda.alloc` function, which takes the
            buffer size as an argument and returns the device buffer of that
            size. When ``None`` is specified, raw memory allocator will be
            used (i.e., memory pool is disabled).

    """
    global _current_allocator
    if allocator is None:
        allocator = _malloc
    _current_allocator = allocator


class PooledMemory(Memory):

    """Memory allocation for a memory pool.

    The instance of this class is created by memory pool allocator, so user
    should not instantiate it by hand.

    """

    def __init__(self, Chunk chunk, pool):
        self.device = chunk.device
        self.ptr = chunk.ptr
        self.size = chunk.size
        self.pool = pool

    def free(self):
        """Frees the memory buffer and returns it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        cdef Py_ssize_t ptr
        ptr = self.ptr
        if ptr == 0:
            return
        self.ptr = 0
        pool = self.pool()
        if pool is None:
            return

        hooks = None
        # to avoid error at exit
        hooks = memory_hook.get_memory_hooks()
        size = self.size
        if hooks:
            device_id = self.device.id
            pmem_id = id(self)
            hooks_values = hooks.values()  # avoid six for performance
            for hook in hooks_values:
                hook.free_preprocess(device_id=device_id,
                                     mem_size=size,
                                     mem_ptr=ptr,
                                     pmem_id=pmem_id)
            try:
                pool.free(ptr, size)
            finally:
                for hook in hooks_values:
                    hook.free_postprocess(device_id=device_id,
                                          mem_size=size,
                                          mem_ptr=ptr,
                                          pmem_id=pmem_id)
        else:
            pool.free(ptr, size)

    __del__ = free


cdef int _compaction_threshold = 512


cdef _compaction(SingleDeviceMemoryPool pool, size_t stream_ptr, bint free):
    # need self._free_lock
    cdef list arena, new_arena
    cdef set free_list, keep_list
    cdef vector.vector[int]* arena_index
    cdef vector.vector[int] new_index
    cdef size_t index

    new_arena = []
    arena = pool._free[stream_ptr]
    arena_index = &pool._index[stream_ptr]
    for index in range(len(arena)):
        free_list = arena[index]
        if not free_list:
            continue
        if free:
            keep_list = set()
            for chunk in free_list:
                if chunk.prev is not None or chunk.next is not None:
                    keep_list.add(chunk)
            if len(keep_list) == 0:
                continue
            free_list = keep_list

        new_index.push_back(arena_index.at(index))
        new_arena.append(free_list)
    arena_index.swap(new_index)
    arena[:] = new_arena



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
        self._in_use = {}
        self._free = {}
        self._allocator = allocator
        self._weakref = weakref.ref(self)
        self._device_id = device.get_device_id()
        self._free_lock = rlock.create_fastrlock()
        self._in_use_lock = rlock.create_fastrlock()

    cpdef Py_ssize_t _round_size(self, Py_ssize_t size):
        """Round up the memory size to fit memory alignment of cudaMalloc."""
        unit = self._allocation_unit_size
        return ((size + unit - 1) // unit) * unit

    cpdef int _bin_index_from_size(self, Py_ssize_t size):
        """Get appropriate bins index from the memory size"""
        unit = self._allocation_unit_size
        return (size - 1) // unit

    cpdef list _arena(self, size_t stream_ptr):
        """Get appropriate arena (list of bins) of a given stream"""
        if stream_ptr not in self._free:
            self._free[stream_ptr] = []
        return self._free[stream_ptr]

    cdef vector.vector[int]* _arena_index(self, size_t stream_ptr):
        """Get appropriate arena sparse index of a given stream"""
        return &self._index[stream_ptr]

    cpdef _append_to_free_list(self, Py_ssize_t size, chunk,
                               size_t stream_ptr):
        cdef int index, bin_index
        cdef list arena
        cdef set free_list
        cdef vector.vector[int]* arena_index

        bin_index = self._bin_index_from_size(size)
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            arena = self._arena(stream_ptr)
            arena_index = self._arena_index(stream_ptr)
            index = algorithm.lower_bound(
                arena_index.begin(), arena_index.end(),
                bin_index) - arena_index.begin()
            size = <int>arena_index.size()
            if index < size and arena_index.at(index) == bin_index:
                free_list = arena[index]
                if free_list is None:
                    arena[index] = free_list = set()
            else:
                free_list = set()
                arena_index.insert(arena_index.begin() + index, bin_index)
                arena.insert(index, free_list)
            free_list.add(chunk)
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef bint _remove_from_free_list(self, Py_ssize_t size, chunk,
                                      size_t stream_ptr) except *:
        cdef int index, bin_index
        cdef list arena
        cdef set free_list
        cdef vector.vector[int]* arena_index

        bin_index = self._bin_index_from_size(size)
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            arena = self._arena(stream_ptr)
            arena_index = self._arena_index(stream_ptr)
            index = algorithm.lower_bound(
                arena_index.begin(), arena_index.end(),
                bin_index) - arena_index.begin()
            if arena_index.at(index) != bin_index:
                return False
            free_list = arena[index]
            if free_list and chunk in free_list:
                free_list.remove(chunk)
                if len(free_list) == 0:
                    arena[index] = None
                return True
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return False

    cpdef tuple _split(self, Chunk chunk, Py_ssize_t size):
        """Split contiguous block of a larger allocation"""
        cdef Chunk head
        cdef Chunk remaining
        assert chunk.size >= size
        if chunk.size == size:
            return chunk, None
        head = Chunk(chunk.mem, chunk.offset, size, chunk.stream_ptr)
        remaining = Chunk(chunk.mem, chunk.offset + size, chunk.size - size,
                          chunk.stream_ptr)
        if chunk.prev is not None:
            head.prev = chunk.prev
            chunk.prev.next = head
        if chunk.next is not None:
            remaining.next = chunk.next
            chunk.next.prev = remaining
        head.next = remaining
        remaining.prev = head
        return head, remaining

    cpdef Chunk _merge(self, Chunk head, Chunk remaining):
        """Merge previously splitted block (chunk)"""
        assert head.stream_ptr == remaining.stream_ptr
        cdef Chunk merged
        size = head.size + remaining.size
        merged = Chunk(head.mem, head.offset, size, head.stream_ptr)
        if head.prev is not None:
            merged.prev = head.prev
            merged.prev.next = merged
        if remaining.next is not None:
            merged.next = remaining.next
            merged.next.prev = merged
        return merged

    cpdef MemoryPointer _alloc(self, Py_ssize_t rounded_size):
        hooks = memory_hook.get_memory_hooks()
        if hooks:
            memptr = None
            device_id = self._device_id
            hooks_values = hooks.values()  # avoid six for performance
            for hook in hooks_values:
                hook.alloc_preprocess(device_id=device_id,
                                      mem_size=rounded_size)
            try:
                memptr = self._allocator(rounded_size)
            finally:
                for hook in hooks_values:
                    mem_ptr = memptr.ptr if memptr is not None else 0
                    hook.alloc_postprocess(device_id=device_id,
                                           mem_size=rounded_size,
                                           mem_ptr=mem_ptr)
            return memptr
        else:
            return self._allocator(rounded_size)

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        rounded_size = self._round_size(size)
        hooks = memory_hook.get_memory_hooks()
        if hooks:
            memptr = None
            device_id = self._device_id
            hooks_values = hooks.values()  # avoid six for performance
            for hook in hooks_values:
                hook.malloc_preprocess(device_id=device_id,
                                       size=size,
                                       mem_size=rounded_size)
            try:
                memptr = self._malloc(rounded_size)
            finally:
                if memptr is None:
                    mem_ptr = 0
                    pmem_id = 0
                else:
                    mem_ptr = memptr.ptr
                    pmem_id = id(memptr.mem)
                for hook in hooks_values:
                    hook.malloc_postprocess(device_id=device_id,
                                            size=size,
                                            mem_size=rounded_size,
                                            mem_ptr=mem_ptr,
                                            pmem_id=pmem_id)
            return memptr
        else:
            return self._malloc(rounded_size)

    cpdef MemoryPointer _malloc(self, Py_ssize_t size):
        cdef set free_list
        cdef Chunk chunk = None
        cdef Chunk remaining = None
        cdef int bin_index, index, length

        if size == 0:
            return MemoryPointer(Memory(0), 0)

        stream_ptr = stream_module.get_current_stream_ptr()

        bin_index = self._bin_index_from_size(size)
        # find best-fit, or a smallest larger allocation
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            arena = self._arena(stream_ptr)
            arena_index = self._arena_index(stream_ptr)
            index = algorithm.lower_bound(
                arena_index.begin(), arena_index.end(),
                bin_index) - arena_index.begin()
            length = arena_index.size()
            for i in range(index, length):
                free_list = arena[i]
                if free_list is None:
                    continue
                assert len(free_list) > 0
                chunk = free_list.pop()
                if len(free_list) == 0:
                    arena[i] = None
                if i - index >= _compaction_threshold:
                    _compaction(self, stream_ptr, False)
                break
        finally:
            rlock.unlock_fastrlock(self._free_lock)

        if chunk is not None:
            _ensure_context(self._device_id)
            chunk, remaining = self._split(chunk, size)
        else:
            # cudaMalloc if a cache is not found
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
                    try:
                        mem = self._alloc(size).mem
                    except runtime.CUDARuntimeError as e:
                        if e.status != runtime.errorMemoryAllocation:
                            raise
                        else:
                            total = size + self.total_bytes()
                            raise OutOfMemoryError(size, total)
            chunk = Chunk(mem, 0, size, stream_ptr)

        assert chunk.stream_ptr == stream_ptr
        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            self._in_use[chunk.ptr] = chunk
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        if remaining is not None:
            self._append_to_free_list(remaining.size, remaining, stream_ptr)
        pmem = PooledMemory(chunk, self._weakref)
        return MemoryPointer(pmem, 0)

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef set free_list
        cdef Chunk chunk

        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            chunk = self._in_use.pop(ptr, None)
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        if chunk is None:
            raise RuntimeError('Cannot free out-of-pool memory')
        stream_ptr = chunk.stream_ptr

        if chunk.next is not None:
            if self._remove_from_free_list(chunk.next.size, chunk.next,
                                           stream_ptr):
                chunk = self._merge(chunk, chunk.next)

        if chunk.prev is not None:
            if self._remove_from_free_list(chunk.prev.size, chunk.prev,
                                           stream_ptr):
                chunk = self._merge(chunk.prev, chunk)

        self._append_to_free_list(chunk.size, chunk, stream_ptr)

    cpdef free_all_blocks(self, stream=None, stream_ptr=None):
        """Free all **non-split** chunks"""
        cdef set free_list, keep_list
        cdef Chunk chunk
        cdef list arena
        cdef vector.vector[int]* arena_index
        cdef size_t index
        cdef size_t _stream_ptr

        # free blocks in all arenas
        if stream is None and stream_ptr is None:
            rlock.lock_fastrlock(self._free_lock, -1, True)
            try:
                for _stream_ptr in list(self._free.iterkeys()):
                    self.free_all_blocks(stream_ptr=_stream_ptr)
            finally:
                rlock.unlock_fastrlock(self._free_lock)
            return

        # free blocks in the arena of the given stream
        if stream_ptr is not None:
            _stream_ptr = <size_t>stream_ptr
        elif stream is not None:
            _stream_ptr = <size_t>stream.ptr
        else:
            assert(False)
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            if _stream_ptr not in self._free:
                return
            _compaction(self, _stream_ptr, True)
            arena = self._free[_stream_ptr]
            if len(arena) == 0:
                self._index.erase(_stream_ptr)
                del self._free[_stream_ptr]
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        cdef Py_ssize_t n = 0
        cdef set free_list
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            for arena in self._free.itervalues():
                for v in arena:
                    if v is not None:
                        n += len(v)
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return n

    cpdef used_bytes(self):
        cdef Py_ssize_t size = 0
        cdef Chunk chunk
        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            for chunk in self._in_use.itervalues():
                size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        return size

    cpdef free_bytes(self):
        cdef Py_ssize_t size = 0
        cdef set free_list
        cdef Chunk chunk
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            for arena in self._free.itervalues():
                for free_list in arena:
                    if free_list is None:
                        continue
                    for chunk in free_list:
                        size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return size

    cpdef total_bytes(self):
        return self.used_bytes() + self.free_bytes()


cdef class MemoryPool(object):

    """Memory pool for all GPU devices on the host.

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

        Also, the way to use a memory pool of Managed memory (Unified memory)
        as the default allocator is the following code::

            set_allocator(MemoryPool(malloc_managed).malloc)

        Args:
            size (int): Size of the memory buffer to allocate in bytes.

        Returns:
            ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.malloc(size)

    cpdef free_all_blocks(self, stream=None, stream_ptr=None):
        """Release free blocks.

        Args:
            stream (cupy.cuda.Stream): Release free blocks in the arena
                of the given stream. The default releases blocks in all
                arenas. Specify either of `stream` or `stream_ptr`.
            stream_ptr (size_t): Release free blocks in the arena
                of the given stream. The default releases blocks in all
                arenas. Specify either of `stream` or `stream_ptr`.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        mp.free_all_blocks(stream=stream, stream_ptr=stream_ptr)

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

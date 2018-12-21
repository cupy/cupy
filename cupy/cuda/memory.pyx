# distutils: language = c++
cimport cython  # NOQA

import atexit
import collections
import ctypes
import gc
import warnings
import weakref

from cpython cimport pythread
from cython.operator cimport dereference
from fastrlock cimport rlock
from libc.stdint cimport int8_t
from libc.stdint cimport intptr_t
from libcpp cimport algorithm

from cupy.cuda import runtime

from cupy.cuda cimport device
from cupy.cuda cimport device as device_mod
from cupy.cuda cimport memory_hook
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module


cdef bint _exit_mode = False


@atexit.register
def _exit():
    _exit_mode = True


class OutOfMemoryError(MemoryError):

    def __init__(self, size, total):
        msg = 'out of memory to allocate %d bytes ' \
              '(total %d bytes)' % (size, total)
        super(OutOfMemoryError, self).__init__(msg)


@cython.no_gc
cdef class BaseMemory:
    """Memory on a CUDA device.

    Attributes:
        ~Memory.ptr (int): Pointer to the place within the buffer.
        ~Memory.size (int): Size of the memory allocation in bytes.
        ~Memory.device (~cupy.cuda.Device): Device whose memory the pointer
            refers to.
    """

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr

    @property
    def device(self):
        return device.Device(self.device_id)


@cython.no_gc
cdef class Memory(BaseMemory):
    """Memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.
    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device_id = device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.malloc(size)

    def __dealloc__(self):
        if self.ptr:
            runtime.free(self.ptr)


cdef class UnownedMemory(BaseMemory):
    """CUDA memory that is not owned by CuPy.

    Args:
        ptr (int): Pointer to the buffer.
        size (int): Size of the buffer.
        owner (object): Reference to the owner object to keep the memory
            alive.
        device_id (int): CUDA device ID of the buffer. If omitted, the device
            associated to the pointer is retrieved.
    """

    cdef:
        readonly object _owner

    def __init__(self, size_t ptr, Py_ssize_t size, object owner,
                 int device_id=-1):
        cdef runtime.PointerAttributes ptr_attrs
        if device_id < 0:
            ptr_attrs = runtime.pointerGetAttributes(ptr)
            device_id = ptr_attrs.device
        self.size = size
        self.device_id = device_id
        self.ptr = ptr
        self._owner = owner


@cython.no_gc
cdef class ManagedMemory(BaseMemory):
    """Managed memory (Unified memory) allocation on a CUDA device.

    This class provides an RAII interface of the CUDA managed memory
    allocation.

    Args:
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device_id = device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.mallocManaged(size)

    def prefetch(self, stream):
        """(experimental) Prefetch memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream.
        """
        runtime.memPrefetchAsync(self.ptr, self.size, self.device_id,
                                 stream.ptr)

    def advise(self, int advise, device_mod.Device device):
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


@cython.final
cdef class _Chunk:

    """A chunk points to a device memory.

    A chunk might be a splitted memory block from a larger allocation.
    The prev/next pointers contruct a doubly-linked list of memory addresses
    sorted by base address that must be contiguous.

    Args:
        mem (~cupy.cuda.Memory): The device memory buffer.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        stream_ptr (size_t): Raw stream handle of cupy.cuda.Stream

    Attributes:
        mem (Memory): The device memory buffer.
        ptr (size_t): Memory address.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        prev (Chunk): prev memory pointer if split from a larger allocation
        next (Chunk): next memory pointer if split from a larger allocation
        stream_ptr (size_t): Raw stream handle of cupy.cuda.Stream
    """

    cdef:
        readonly BaseMemory mem
        readonly Py_ssize_t offset
        readonly Py_ssize_t size
        readonly size_t stream_ptr
        public _Chunk prev
        public _Chunk next

    def __init__(self, *args):
        # For debug
        mem, offset, size, stream_ptr = args
        self._init(mem, offset, size, stream_ptr)

    cdef _init(self, BaseMemory mem, Py_ssize_t offset,
               Py_ssize_t size, Py_ssize_t stream_ptr):
        assert mem.ptr > 0 or offset == 0
        self.mem = mem
        self.offset = offset
        self.size = size
        self.stream_ptr = stream_ptr

    cpdef size_t ptr(self):
        return self.mem.ptr + self.offset

    cpdef _Chunk split(self, Py_ssize_t size):
        """Split contiguous block of a larger allocation"""
        cdef _Chunk remaining
        assert self.size >= size
        if self.size == size:
            return None
        remaining = _Chunk.__new__(_Chunk)
        remaining._init(self.mem, self.offset + size, self.size - size,
                        self.stream_ptr)
        self.size = size

        if self.next is not None:
            remaining.next = self.next
            remaining.next.prev = remaining
        self.next = remaining
        remaining.prev = self
        return remaining

    cpdef merge(self, _Chunk remaining):
        """Merge previously splitted block (chunk)"""
        assert self.stream_ptr == remaining.stream_ptr
        self.size += remaining.size
        self.next = remaining.next
        if remaining.next is not None:
            self.next.prev = self


cdef class MemoryPointer:
    """Pointer to a point on a device memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (~cupy.cuda.BaseMemory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        ~MemoryPointer.device (~cupy.cuda.Device): Device whose memory the
            pointer refers to.
        ~MemoryPointer.mem (~cupy.cuda.BaseMemory): The device memory buffer.
        ~MemoryPointer.ptr (size_t): Pointer to the place within the buffer.
    """

    def __init__(self, BaseMemory mem, Py_ssize_t offset):
        self._init(mem, offset)

    cdef _init(self, BaseMemory mem, Py_ssize_t offset):
        assert mem.ptr > 0 or offset == 0
        self.ptr = mem.ptr + offset
        self.device_id = mem.device_id
        self.mem = mem

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr

    @property
    def device(self):
        return device.Device(self.device_id)

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
            _set_peer_access(src.device_id, self.device_id)
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
            _set_peer_access(src.device_id, self.device_id)
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

    Read more at: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#axzz4qygc1Ry1  # NOQA

    Args:
        size (int): Size of the memory allocation in bytes.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.
    """
    mem = ManagedMemory(size)
    return MemoryPointer(mem, 0)


cdef object _current_allocator = _malloc


cpdef MemoryPointer alloc(size):
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


@cython.final
@cython.no_gc
cdef class PooledMemory(BaseMemory):

    """Memory allocation for a memory pool.

    The instance of this class is created by memory pool allocator, so user
    should not instantiate it by hand.

    """

    cdef:
        readonly object pool

    def __init__(self, _Chunk chunk, pool):
        self.ptr = chunk.ptr()
        self.size = chunk.size
        self.device_id = chunk.mem.device_id
        self.pool = pool

    cpdef free(self):
        """Frees the memory buffer and returns it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        cdef size_t ptr
        ptr = self.ptr
        if ptr == 0:
            return
        self.ptr = 0
        pool = self.pool()
        if pool is None:
            return

        size = self.size
        if memory_hook._has_memory_hooks():
            hooks = memory_hook.get_memory_hooks()
            if hooks:
                device_id = self.device_id
                pmem_id = id(self)

                # avoid six for performance
                hooks_values = hooks.values()
                for hook in hooks_values:
                    hook.free_preprocess(device_id=device_id,
                                         mem_size=size,
                                         mem_ptr=ptr,
                                         pmem_id=pmem_id)
                try:
                    (<SingleDeviceMemoryPool>pool).free(ptr, size)
                finally:
                    for hook in hooks_values:
                        hook.free_postprocess(device_id=device_id,
                                              mem_size=size,
                                              mem_ptr=ptr,
                                              pmem_id=pmem_id)
                return
        (<SingleDeviceMemoryPool>pool).free(ptr, size)

    def __dealloc__(self):
        if _exit_mode:
            return  # To avoid error at exit
        self.free()


cdef int _index_compaction_threshold = 512


cdef _compact_index(SingleDeviceMemoryPool pool, size_t stream_ptr, bint free):
    # need self._free_lock
    cdef list arena, new_arena
    cdef set free_list, keep_list
    cdef vector.vector[int]* arena_index
    cdef vector.vector[int] new_index
    cdef size_t index

    if stream_ptr not in pool._free:
        return
    new_arena = []
    arena = pool._free[stream_ptr]
    arena_index = pool._arena_index(stream_ptr)
    for index, free_list in enumerate(arena):
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
    if free and len(new_arena) == 0:
        pool._index.erase(stream_ptr)
        pool._flag.erase(stream_ptr)
        del pool._free[stream_ptr]
    else:
        arena_index.swap(new_index)
        arena[:] = new_arena
        pool._arena_flag(stream_ptr).assign(new_index.size(), <int8_t>1)


cdef object _get_chunk(SingleDeviceMemoryPool pool, Py_ssize_t size,
                       size_t stream_ptr):
    # need self._free_lock
    cdef set free_list
    cdef int i, index, length
    cdef _Chunk chunk
    cdef int bin_index = _bin_index_from_size(size)
    cdef list arena = pool._arena(stream_ptr)
    a_index = pool._arena_index(stream_ptr)
    a_flag = pool._arena_flag(stream_ptr)
    index = algorithm.lower_bound(
        a_index.begin(), a_index.end(), bin_index) - a_index.begin()
    length = a_index.size()
    for i in range(index, length):
        if a_flag.at(i) == 0:
            continue
        free_list = arena[i]
        chunk = free_list.pop()
        if len(free_list) == 0:
            dereference(a_flag)[i] = 0
            arena[i] = None
        if i - index >= _index_compaction_threshold:
            _compact_index(pool, stream_ptr, False)
        remaining = chunk.split(size)
        if remaining is not None:
            _append_to_free_list(arena, a_index, a_flag, remaining)
        assert chunk.stream_ptr == stream_ptr
        return chunk
    return None


cdef BaseMemory _try_malloc(SingleDeviceMemoryPool pool, Py_ssize_t size):
    try:
        return pool._alloc(size).mem
    except runtime.CUDARuntimeError as e:
        if e.status != runtime.errorMemoryAllocation:
            raise
        pool.free_all_blocks()
        try:
            return pool._alloc(size).mem
        except runtime.CUDARuntimeError as e:
            if e.status != runtime.errorMemoryAllocation:
                raise
            gc.collect()
            try:
                return pool._alloc(size).mem
            except runtime.CUDARuntimeError as e:
                if e.status != runtime.errorMemoryAllocation:
                    raise
    total = size + pool.total_bytes()
    raise OutOfMemoryError(size, total)


cdef _append_to_free_list(list arena, vector.vector[int]* a_index,
                          vector.vector[int8_t]* a_flag, _Chunk chunk):
    # need self._free_lock
    cdef int index, bin_index
    cdef set free_list
    bin_index = _bin_index_from_size(chunk.size)
    index = algorithm.lower_bound(
        a_index.begin(), a_index.end(), bin_index) - a_index.begin()
    if index < <int>a_index.size() and a_index.at(index) == bin_index:
        free_list = arena[index]
        if free_list is None:
            arena[index] = free_list = set()
    else:
        free_list = set()
        a_index.insert(a_index.begin() + index, bin_index)
        a_flag.insert(a_flag.begin() + index, 0)
        arena.insert(index, free_list)
    free_list.add(chunk)
    dereference(a_flag)[index] = 1


cdef bint _remove_from_free_list(list arena, vector.vector[int]* a_index,
                                 vector.vector[int8_t]* a_flag,
                                 _Chunk chunk) except *:
    """Removes the chunk from the free list (need self._free_lock).

    Returns:
        bool: ``True`` if the chunk can successfully be removed from
            the free list. ``False`` otherwise (e.g., the chunk could not
            be found in the free list as the chunk is allocated.)
    """

    cdef int index, bin_index
    cdef set free_list

    bin_index = _bin_index_from_size(chunk.size)
    if a_index.size() == 0:
        return False
    index = algorithm.lower_bound(
        a_index.begin(), a_index.end(), bin_index) - a_index.begin()
    if index == a_index.size():
        # Bin does not exist for the given chunk size.
        return False
    if a_index.at(index) != bin_index or a_flag.at(index) == 0:
        return False
    free_list = arena[index]
    if chunk in free_list:
        free_list.remove(chunk)
        if len(free_list) == 0:
            arena[index] = None
            dereference(a_flag)[index] = 0
        return True
    return False


# cudaMalloc() is aligned to at least 512 bytes
# cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
DEF ALLOCATION_UNIT_SIZE = 512
# for test
_allocation_unit_size = ALLOCATION_UNIT_SIZE


cpdef Py_ssize_t _round_size(Py_ssize_t size):
    """Rounds up the memory size to fit memory alignment of cudaMalloc."""
    # avoid 0 div checking
    size = (size + ALLOCATION_UNIT_SIZE - 1) // ALLOCATION_UNIT_SIZE
    return size * ALLOCATION_UNIT_SIZE

cpdef int _bin_index_from_size(Py_ssize_t size):
    """Returns appropriate bins index from the memory size."""
    # avoid 0 div checking
    return (size - 1) // ALLOCATION_UNIT_SIZE


@cython.final
cdef class SingleDeviceMemoryPool:
    """Memory pool implementation for single device.

    - The allocator attempts to find the smallest cached block that will fit
      the requested size. If the block is larger than the requested size,
      it may be split. If no block is found, the allocator will delegate to
      cudaMalloc.
    - If the cudaMalloc fails, the allocator will free all cached blocks that
      are not split and retry the allocation.
    """

    cdef:
        object _allocator

        # Map from memory pointer of the chunk (size_t) to the corresponding
        # Chunk object. All chunks currently allocated to the application from
        # this pool are stored.
        # `_in_use_lock` must be acquired to access.
        dict _in_use

        # Map from stream pointer (int) to its arena (list) for the stream.
        # `_free_lock` must be acquired to access.
        dict _free

        object __weakref__
        object _weakref
        object _free_lock
        object _in_use_lock
        readonly int _device_id

        # Map from stream pointer to its arena index.
        # `_free_lock` must be acquired to access.
        map.map[size_t, vector.vector[int]] _index
        map.map[size_t, vector.vector[int8_t]] _flag

    def __init__(self, allocator=_malloc):
        self._in_use = {}
        self._free = {}
        self._allocator = allocator
        self._weakref = weakref.ref(self)
        self._device_id = device.get_device_id()
        self._free_lock = rlock.create_fastrlock()
        self._in_use_lock = rlock.create_fastrlock()

    cpdef list _arena(self, size_t stream_ptr):
        """Returns appropriate arena (list of bins) of a given stream.

        All free chunks in the stream belong to one of the bin in the arena.

        Caller is responsible to acquire `_free_lock`.
        """
        ret = self._free.get(stream_ptr, None)
        if ret is None:
            self._free[stream_ptr] = ret = []
        return ret

    cdef inline vector.vector[int]* _arena_index(self, size_t stream_ptr):
        """Returns appropriate arena sparse index of a given stream.

        Each element of the returned vector is an index value of the arena
        for the stream. The k-th element of the arena index is the bin index
        of the arena. For example, when the arena index is `[1, 3]`, it means
        that the arena has 2 bins, and `arena[0]` is for bin index 1 and
        `arena[1]` is for bin index 3.

        Caller is responsible to acquire `_free_lock`.
        """
        return &self._index[stream_ptr]

    cdef vector.vector[int8_t]* _arena_flag(self, size_t stream_ptr):
        """Returns appropriate arena used flag list of a given stream.

        Caller is responsible to acquire `_free_lock`.
        """
        return &self._flag[stream_ptr]

    cpdef MemoryPointer _alloc(self, Py_ssize_t rounded_size):
        if memory_hook._has_memory_hooks():
            hooks = memory_hook.get_memory_hooks()
            if hooks:
                memptr = None
                device_id = self._device_id
                # avoid six for performance
                hooks_values = hooks.values()
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
        return self._allocator(rounded_size)

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        rounded_size = _round_size(size)
        if memory_hook._has_memory_hooks():
            hooks = memory_hook.get_memory_hooks()
            if hooks:
                memptr = None
                device_id = self._device_id
                # avoid six for performance
                hooks_values = hooks.values()
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
        return self._malloc(rounded_size)

    cpdef MemoryPointer _malloc(self, Py_ssize_t size):
        cdef _Chunk chunk
        cdef long current_thread
        cdef BaseMemory mem
        cdef MemoryPointer ret
        if size == 0:
            return MemoryPointer(Memory(0), 0)

        current_thread = pythread.PyThread_get_thread_ident()
        stream_ptr = stream_module.get_current_stream_ptr()

        # find best-fit, or a smallest larger allocation
        rlock.lock_fastrlock(self._free_lock, current_thread, True)
        try:
            chunk = _get_chunk(self, size, stream_ptr)
        finally:
            rlock.unlock_fastrlock(self._free_lock)

        if chunk is None:
            mem = _try_malloc(self, size)
            chunk = _Chunk.__new__(_Chunk)
            # cudaMalloc if a cache is not found
            chunk._init(mem, 0, size, stream_ptr)

        rlock.lock_fastrlock(self._in_use_lock, current_thread, True)
        try:
            self._in_use[chunk.ptr()] = chunk
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        pmem = PooledMemory(chunk, self._weakref)
        return MemoryPointer(pmem, 0)

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef _Chunk chunk, c
        cdef long current_thread = pythread.PyThread_get_thread_ident()

        rlock.lock_fastrlock(self._in_use_lock, current_thread, True)
        try:
            chunk = self._in_use.pop(ptr)
        except KeyError:
            raise RuntimeError('Cannot free out-of-pool memory')
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        stream_ptr = chunk.stream_ptr

        rlock.lock_fastrlock(self._free_lock, current_thread, True)
        try:
            arena = self._arena(stream_ptr)
            a_index = self._arena_index(stream_ptr)
            a_flag = self._arena_flag(stream_ptr)

            c = chunk.next
            if c is not None and _remove_from_free_list(arena, a_index,
                                                        a_flag, c):
                chunk.merge(c)

            c = chunk.prev
            if c is not None and _remove_from_free_list(arena, a_index,
                                                        a_flag, c):
                c.merge(chunk)
                chunk = c

            _append_to_free_list(arena, a_index, a_flag, chunk)
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef free_all_blocks(self, stream=None):
        """Free all **non-split** chunks"""
        cdef size_t stream_ptr

        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            # free blocks in all arenas
            if stream is None:
                for stream_ptr in list(self._free.iterkeys()):
                    _compact_index(self, stream_ptr, True)
            else:
                _compact_index(self, stream.ptr, True)
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
        cdef _Chunk chunk
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
        cdef _Chunk chunk
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

    cpdef free_all_blocks(self, stream=None):
        """Release free blocks.

        Args:
            stream (cupy.cuda.Stream): Release free blocks in the arena
                of the given stream. The default releases blocks in all
                arenas.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        mp.free_all_blocks(stream=stream)

    cpdef free_all_free(self):
        """(Deprecated) Use :meth:`free_all_blocks` instead."""
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



ctypedef void*(*malloc_func)(void*, size_t, int)
ctypedef void(*free_func)(void*, void*, int)


cpdef size_t _call_malloc(intptr_t param_ptr, intptr_t malloc_ptr,
                          Py_ssize_t size, int device_id):
    return <size_t>((<malloc_func>malloc_ptr)(<void*>param_ptr, size,
                                              device_id))


cpdef void _call_free(intptr_t param_ptr, intptr_t free_ptr, intptr_t ptr,
                      int device_id):
    (<free_func>free_ptr)(<void*>param_ptr, <void*>ptr, device_id)


@cython.no_gc
cdef class ExternalAllocatorMemory(BaseMemory):

    def __init__(
            self, Py_ssize_t size, intptr_t param_ptr, intptr_t malloc_ptr,
            intptr_t free_ptr, int device_id):
        self._param_ptr = param_ptr
        self._free_ptr = free_ptr
        self.device_id = device_id
        self.ptr = 0
        if size > 0:
            self.ptr = _call_malloc(param_ptr, malloc_ptr, size, device_id)


    def __dealloc__(self):
        if self.ptr:
            _call_free(self._param_ptr, self._free_ptr, self.ptr,
                       self.device_id)


cdef class ExternalAllocator:

    """Allocator with function pointers to allocation routines.

    This allocator keeps raw pointers to a *param* object along with functions
    pointers to *malloc* and *free*, delegating the actual allocation to
    external sources while only handling the timing of the resource allocation
    and deallocation.

    *malloc* should follow the signature ``void*(*malloc)(void*, size_t, int)``
    returning the pointer to the allocated memory given the pointer to
    *param*, the number of bytes to allocate and the device id on which the
    allocation should take place.

    Similarly, *free* should follow the signature
    ``void(*free)(void*, void*, int)`` with no return, taking the pointer to
    *param*, the pointer to the allocated memory and the device id on which the
    memory was allocated.

    Args:
        param_ptr (int): Address of *param*.
        malloc_ptr (int): Address of *malloc*.
        free_ptr (int): Address of *free*.

    """

    def __init__(self, param_ptr, malloc_ptr, free_ptr):
        self._param_ptr = param_ptr
        self._malloc_ptr = malloc_ptr
        self._free_ptr = free_ptr

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        mem = ExternalAllocatorMemory(size, self._param_ptr, self._malloc_ptr,
                                      self._free_ptr, device.get_device_id())
        return MemoryPointer(mem, 0)

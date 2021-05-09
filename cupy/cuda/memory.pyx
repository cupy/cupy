# distutils: language = c++
cimport cpython  # NOQA
cimport cython  # NOQA

import atexit
import collections
import ctypes
import gc
import os
import threading
import warnings
import weakref

from cupy_backends.cuda.api.runtime import CUDARuntimeError
from cupy._core import syncdetect

from fastrlock cimport rlock
from libc.stdint cimport int8_t
from libc.stdint cimport intptr_t
from libcpp cimport algorithm

from cupy.cuda cimport device
from cupy.cuda cimport memory_hook
from cupy.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport runtime

from cupy import _util


cdef bint _exit_mode = False


@atexit.register
def _exit():
    _exit_mode = True


class OutOfMemoryError(MemoryError):
    """Out-of-memory error.

    Args:
        size (int): Size of memory about to be allocated.
        total (int): Size of memory successfully allocated so far.
        limit (int): Allocation limit.
    """

    def __init__(self, size, total, limit=0):
        self._size = size
        self._total = total
        self._limit = limit

        if limit == 0:
            msg = 'Out of memory allocating {:,} bytes'.format(size)
            if total != -1:
                msg += ' (allocated so far: {:,} bytes).'.format(total)
        else:
            msg = (
                'Out of memory allocating {:,} bytes '
                '(allocated so far: {:,} bytes, '
                'limit set to: {:,} bytes).'.format(size, total, limit))
        super(OutOfMemoryError, self).__init__(msg)

    def __reduce__(self):
        return (type(self), (self._size, self._total, self._limit))


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

    def __init__(self, size_t size):
        self.size = size
        self.device_id = device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.malloc(size)

    def __dealloc__(self):
        # Note: Cannot raise in the destructor! (cython/cython#1613)
        if self.ptr:
            runtime.free(self.ptr)


cdef inline void is_async_alloc_supported(int device_id) except*:
    if CUDA_VERSION < 11020:
        raise RuntimeError("memory_async is supported since CUDA 11.2")
    if runtime._is_hip_environment:
        raise RuntimeError('HIP does not support memory_async')
    cdef int dev_id
    cdef list support
    try:
        is_supported = _thread_local.device_support_async_alloc[device_id]
    except AttributeError:
        support = [runtime.deviceGetAttribute(
            runtime.cudaDevAttrMemoryPoolsSupported, dev_id)
            for dev_id in range(runtime.getDeviceCount())]
        _thread_local.device_support_async_alloc = support
        is_supported = support[device_id]
    if not is_supported:
        raise RuntimeError('Device {} does not support '
                           'malloc_async'.format(device_id))


@cython.no_gc
cdef class MemoryAsync(BaseMemory):
    """Asynchronous memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.
        stream (intptr_t): Pointer to the stream on which the memory is
            allocated and freed.
    """
    cdef:
        readonly intptr_t stream

    def __init__(self, size_t size, intptr_t stream):
        self.size = size
        self.device_id = device.get_device_id()
        # The stream is allowed to be destroyed before the memory is freed, so
        # we don't need to hold a reference to the stream.
        self.stream = stream
        is_async_alloc_supported(self.device_id)
        if size > 0:
            self.ptr = runtime.mallocAsync(size, stream)

    def __dealloc__(self):
        # Free is attempted in the following order until success:
        # 1. Free on the stream on which this memory was allocated
        # 2. Free on the current stream, as self.stream is likely
        #    destroyed by now. To enable this we trust the user
        #    has established a correct stream order.
        # 3. Free synchronously (unlikely to happen)
        # Note: Cannot raise in the destructor! (cython/cython#1613)

        cdef intptr_t curr_stream = self.stream
        cdef tuple ok_errors = (runtime.errorInvalidResourceHandle,
                                runtime.errorContextIsDestroyed,)

        if self.ptr:
            try:
                runtime.freeAsync(self.ptr, curr_stream)
            except CUDARuntimeError as e:
                if e.status not in ok_errors:
                    raise
                try:
                    curr_stream = stream_module.get_current_stream_ptr()
                    runtime.freeAsync(self.ptr, curr_stream)
                except CUDARuntimeError as e:
                    if e.status not in ok_errors:
                        raise
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

    def __init__(self, intptr_t ptr, size_t size, object owner,
                 int device_id=-1):
        cdef runtime.PointerAttributes ptr_attrs
        # ptr=0 for 0-size arrays from __cuda_array_interface__ v2:
        # we need a valid device id as null ptr can't be looked up
        if device_id < 0:
            if ptr == 0:
                raise RuntimeError('UnownedMemory requires explicit'
                                   ' device ID for a null pointer.')
            # Initialize a context to workaround a bug in CUDA 10.2+. (#3991)
            runtime._ensure_context()
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

    def __init__(self, size_t size):
        if runtime._is_hip_environment:
            raise RuntimeError('HIP does not support managed memory')
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

    def advise(self, int advise, device.Device dev):
        """(experimental) Advise about the usage of this memory.

        Args:
            advics (int): Advise to be applied for this memory.
            dev (cupy.cuda.Device): Device to apply the advice for.

        """
        runtime.memAdvise(self.ptr, self.size, advise, dev.id)

    def __dealloc__(self):
        # Note: Cannot raise in the destructor! (cython/cython#1613)
        if self.ptr:
            runtime.free(self.ptr)


cdef set _peer_access_checked = set()


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
        stream_ident (intptr_t): Value to uniquely identify the stream.

    Attributes:
        mem (Memory): The device memory buffer.
        ptr (int): Memory address.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        prev (Chunk): prev memory pointer if split from a larger allocation
        next (Chunk): next memory pointer if split from a larger allocation
        stream_ident (intptr_t): Value to uniquely identify the stream.
    """

    cdef:
        readonly BaseMemory mem
        readonly ptrdiff_t offset
        readonly size_t size
        readonly intptr_t stream_ident
        public _Chunk prev
        public _Chunk next

    def __init__(self, *args):
        # For debug
        mem, offset, size, stream_ident = args
        self._init(mem, offset, size, stream_ident)

    cdef _init(self, BaseMemory mem, ptrdiff_t offset,
               size_t size, intptr_t stream_ident):
        assert mem.ptr != 0 or offset == 0
        self.mem = mem
        self.offset = offset
        self.size = size
        self.stream_ident = stream_ident

    cpdef intptr_t ptr(self):
        return self.mem.ptr + self.offset

    cpdef _Chunk split(self, size_t size):
        """Split contiguous block of a larger allocation"""
        cdef _Chunk remaining
        assert self.size >= size
        if self.size == size:
            return None
        remaining = _Chunk.__new__(_Chunk)
        remaining._init(self.mem, self.offset + size, self.size - size,
                        self.stream_ident)
        self.size = size

        if self.next is not None:
            remaining.next = self.next
            remaining.next.prev = remaining
        self.next = remaining
        remaining.prev = self
        return remaining

    cpdef merge(self, _Chunk remaining):
        """Merge previously splitted block (chunk)"""
        assert self.stream_ident == remaining.stream_ident
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
        ~MemoryPointer.ptr (int): Pointer to the place within the buffer.
    """

    def __init__(self, BaseMemory mem, ptrdiff_t offset):
        self._init(mem, offset)

    cdef _init(self, BaseMemory mem, ptrdiff_t offset):
        assert mem.ptr != 0 or offset == 0
        self.ptr = mem.ptr + offset
        self.device_id = mem.device_id
        self.mem = mem

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr

    def __repr__(self):
        return '<{} 0x{:x} device={} mem={!r}>'.format(
            self.__class__.__name__,
            self.ptr, self.device_id, self.mem)

    @property
    def device(self):
        return device.Device(self.device_id)

    def __add__(x, y):
        """Adds an offset to the pointer."""
        cdef MemoryPointer self
        cdef ptrdiff_t offset
        if isinstance(x, MemoryPointer):
            self = x
            offset = <ptrdiff_t?>y
        else:
            self = <MemoryPointer?>y
            offset = <ptrdiff_t?>x
        assert self.ptr != 0 or offset == 0
        return MemoryPointer(self.mem,
                             self.ptr - self.mem.ptr + offset)

    def __iadd__(self, ptrdiff_t offset):
        """Adds an offset to the pointer in place."""
        assert self.ptr != 0 or offset == 0
        self.ptr += offset
        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, ptrdiff_t offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    cpdef copy_from_device(self, MemoryPointer src, size_t size):
        """Copies a memory sequence from a (possibly different) device.

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.

        .. warning::

            This function always uses the legacy default stream and does not
            honor the current stream. Use `copy_from_device_async` instead
            if you are using streams in your code, or have PTDS enabled.

        """
        if size > 0:
            MemoryPointer._set_peer_access(src.device_id, self.device_id)
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
            MemoryPointer._set_peer_access(src.device_id, self.device_id)
            runtime.memcpyAsync(self.ptr, src.ptr, size,
                                runtime.memcpyDefault, stream_ptr)

    cpdef copy_from_host(self, mem, size_t size):
        """Copies a memory sequence from the host memory.

        Args:
            mem (int or ctypes.c_void_p): Source memory pointer.
            size (int): Size of the sequence in bytes.

        .. warning::

            This function always uses the legacy default stream and does not
            honor the current stream. Use `copy_from_host_async` instead
            if you are using streams in your code, or have PTDS enabled.

        """
        if size > 0:
            ptr = mem if isinstance(mem, int) else mem.value
            runtime.memcpy(self.ptr, ptr, size,
                           runtime.memcpyHostToDevice)

    cpdef copy_from_host_async(self, mem, size_t size, stream=None):
        """Copies a memory sequence from the host memory asynchronously.

        Args:
            mem (int or ctypes.c_void_p): Source memory pointer. It must point
                to pinned memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

        """
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        if size > 0:
            ptr = mem if isinstance(mem, int) else mem.value
            runtime.memcpyAsync(self.ptr, ptr, size,
                                runtime.memcpyHostToDevice, stream_ptr)

    cpdef copy_from(self, mem, size_t size):
        """Copies a memory sequence from a (possibly different) device or host.

        This function is a useful interface that selects appropriate one from
        :meth:`~cupy.cuda.MemoryPointer.copy_from_device` and
        :meth:`~cupy.cuda.MemoryPointer.copy_from_host`.

        Args:
            mem (int or ctypes.c_void_p or cupy.cuda.MemoryPointer):
                Source memory pointer.
            size (int): Size of the sequence in bytes.

        .. warning::

            This function always uses the legacy default stream and does not
            honor the current stream. Use `copy_from_async` instead
            if you are using streams in your code, or have PTDS enabled.

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
            mem (int or ctypes.c_void_p or cupy.cuda.MemoryPointer):
                Source memory pointer.
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
            mem (int or ctypes.c_void_p): Target memory pointer.
            size (int): Size of the sequence in bytes.

        .. warning::

            This function always uses the legacy default stream and does not
            honor the current stream. Use `copy_to_host_async` instead
            if you are using streams in your code, or have PTDS enabled.

        """
        if size > 0:
            ptr = mem if isinstance(mem, int) else mem.value
            runtime.memcpy(ptr, self.ptr, size,
                           runtime.memcpyDeviceToHost)

    cpdef copy_to_host_async(self, mem, size_t size, stream=None):
        """Copies a memory sequence to the host memory asynchronously.

        Args:
            mem (int or ctypes.c_void_p): Target memory pointer. It must point
                to pinned memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.
                The default uses CUDA stream of the current context.

        """
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        if size > 0:
            ptr = mem if isinstance(mem, int) else mem.value
            runtime.memcpyAsync(ptr, self.ptr, size,
                                runtime.memcpyDeviceToHost, stream_ptr)

    cpdef memset(self, int value, size_t size):
        """Fills a memory sequence by constant byte value.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.

        .. warning::

            This function always uses the legacy default stream and does not
            honor the current stream. Use `memset_async` instead
            if you are using streams in your code, or have PTDS enabled.

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

    @staticmethod
    cdef _set_peer_access(int device, int peer):
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
        # peer access could already be set by external libraries at this point
        except CUDARuntimeError as e:
            if e.status != runtime.errorPeerAccessAlreadyEnabled:
                raise
        finally:
            runtime.setDevice(current)


# cpdef because unit-tested
cpdef MemoryPointer _malloc(size_t size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_async(size_t size):
    """(Experimental) Allocate memory from Stream Ordered Memory Allocator.

    This method can be used as a CuPy memory allocator. The simplest way to
    use CUDA's Stream Ordered Memory Allocator as the default allocator is
    the following code::

        set_allocator(malloc_async)

    Using this feature requires CUDA >= 11.2 with a supported GPU and platform.
    If it is not supported, an error will be raised.

    The current CuPy stream is used to allocate/free the memory.

    Args:
        size (int): Size of the memory allocation in bytes.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

    .. warning::
        This feature is currently experimental and subject to change.

    .. seealso:: `Stream Ordered Memory Allocator`_

    .. _Stream Ordered Memory Allocator:
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
    """
    cdef intptr_t stream_ptr
    stream_ptr = stream_module.get_current_stream_ptr()
    mem = MemoryAsync(size, stream_ptr)
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_managed(size_t size):
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
cdef object _thread_local = threading.local()


def _get_thread_local_allocator():
    try:
        allocator = _thread_local.allocator
    except AttributeError:
        allocator = _thread_local.allocator = None
    return allocator


def _set_thread_local_allocator(allocator):
    _thread_local.allocator = allocator


cdef inline intptr_t _get_stream_identifier(intptr_t stream_ptr):
    # When PTDS is enabled, return an ID to uniquely identify the default
    # stream for each thread. (#5069)
    if stream_ptr != runtime.streamPerThread:
        return stream_ptr

    cpdef intptr_t tid
    try:
        tid = _thread_local._tid
    except AttributeError:
        _thread_local._tid_obj = tid_obj = object()
        _thread_local._tid = tid = id(tid_obj)
    return -tid


cpdef MemoryPointer alloc(size):
    """Calls the current allocator.

    Use :func:`~cupy.cuda.set_allocator` to change the current allocator.

    Args:
        size (int): Size of the memory allocation.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

    """
    return get_allocator()(size)


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
    if getattr(_thread_local, 'allocator', None) is not None:
        raise ValueError('Can\'t change the global allocator inside '
                         '`using_allocator` context manager')
    if allocator is malloc_async:
        _util.experimental('cupy.cuda.malloc_async')
    _current_allocator = allocator


cpdef get_allocator():
    """Returns the current allocator for GPU memory.

    Returns:
        function: CuPy memory allocator.
    """
    try:
        allocator = _thread_local.allocator
    except AttributeError:
        _thread_local.allocator = allocator = None
    if allocator is None:
        return _current_allocator
    else:
        return allocator


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
        self._init(chunk, pool)

    cdef _init(self, _Chunk chunk, pool):
        self.ptr = chunk.ptr()
        self.size = chunk.size
        self.device_id = chunk.mem.device_id
        self.pool = pool

    cpdef free(self):
        """Frees the memory buffer and returns it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        cdef intptr_t ptr
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

                for hook in hooks.values():
                    hook.free_preprocess(device_id=device_id,
                                         mem_size=size,
                                         mem_ptr=ptr,
                                         pmem_id=pmem_id)
                try:
                    (<SingleDeviceMemoryPool>pool).free(ptr, size)
                finally:
                    for hook in hooks.values():
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


cdef size_t _index_compaction_threshold = 512


# cudaMalloc() is aligned to at least 512 bytes
# cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
DEF ALLOCATION_UNIT_SIZE = 512
# for test
_allocation_unit_size = ALLOCATION_UNIT_SIZE


cpdef inline size_t _round_size(size_t size):
    """Rounds up the memory size to fit memory alignment of cudaMalloc."""
    # avoid 0 div checking
    size = (size + ALLOCATION_UNIT_SIZE - 1) // ALLOCATION_UNIT_SIZE
    return size * ALLOCATION_UNIT_SIZE

cpdef size_t _bin_index_from_size(size_t size):
    """Returns appropriate bins index from the memory size."""
    # avoid 0 div checking
    return (size - 1) // ALLOCATION_UNIT_SIZE


cdef _gc_isenabled = gc.isenabled
cdef _gc_disable = gc.disable
cdef _gc_enable = gc.enable


cdef bint _lock_no_gc(lock):
    """Lock to ensure single thread execution and no garbage collection.

    Returns:
        bool: Whether GC is disabled.
    """
    rlock.lock_fastrlock(lock, -1, True)

    # This function may be called from the context of finalizer
    # (e.g., `__dealloc__` of PooledMemory class).
    # If the process is going to be terminated, the module itself may
    # already been unavailable.
    if not _exit_mode and _gc_isenabled():
        _gc_disable()
        return True
    return False


cdef _unlock_no_gc(lock, bint gc_mode):
    if gc_mode:
        _gc_enable()
    rlock.unlock_fastrlock(lock)


cdef class LockAndNoGc:
    """A context manager that ensures single-thread execution
    and no garbage collection in the wrapped code.
    The purpose of disabling GC is to prevent unexpected recursion.
    See gh-2074 for details.
    """

    cdef object _lock
    cdef bint _gc

    def __cinit__(self, lock):
        self._lock = lock

    def __enter__(self):
        self._gc = _lock_no_gc(self._lock)

    def __exit__(self, t, v, tb):
        _unlock_no_gc(self._lock, self._gc)


@cython.final
cdef class _Arena:

    cdef:
        # `_free_lock` must be acquired to access it.
        list _free
        # `_free_lock` must be acquired to access it.
        vector.vector[size_t] _index
        # `_free_lock` must be acquired to access it.
        vector.vector[int8_t] _flag

    def __init__(self):
        self._free = []

    cdef append_to_free_list(self, _Chunk chunk):
        # need self._free_lock
        cdef size_t index, bin_index
        cdef set free_list
        cdef vector.vector[size_t].iterator it

        bin_index = _bin_index_from_size(chunk.size)
        it = algorithm.lower_bound(
            self._index.begin(), self._index.end(), bin_index)
        index = <size_t>(it - self._index.begin())
        if index < self._index.size() and self._index.at(index) == bin_index:
            free_list = self._free[index]
            if free_list is None:
                self._free[index] = free_list = set()
        else:
            free_list = set()
            self._index.insert(self._index.begin() + index, bin_index)
            self._flag.insert(self._flag.begin() + index, 0)
            self._free.insert(index, free_list)
        free_list.add(chunk)
        self._flag[index] = 1

    cdef bint remove_from_free_list(self, _Chunk chunk):
        """Removes the chunk from the free list (need self._free_lock).

        Returns:
            bool: ``True`` if the chunk can successfully be removed from
                the free list. ``False`` otherwise (e.g., the chunk could not
                be found in the free list as the chunk is allocated.)
        """

        cdef size_t index, bin_index
        cdef set free_list
        cdef vector.vector[size_t].iterator it

        bin_index = _bin_index_from_size(chunk.size)
        if self._index.size() == 0:
            return False
        it = algorithm.lower_bound(
            self._index.begin(), self._index.end(), bin_index)
        index = <size_t>(it - self._index.begin())
        if index == self._index.size():
            # Bin does not exist for the given chunk size.
            return False
        if self._index.at(index) != bin_index or self._flag.at(index) == 0:
            return False
        free_list = self._free[index]
        if chunk in free_list:
            free_list.remove(chunk)
            if len(free_list) == 0:
                self._free[index] = None
                self._flag[index] = 0
            return True
        return False


# cpdef because uint-tested
# module-level function can be inlined
cpdef inline dict _parse_limit_string(limit=None):
    if limit is None:
        limit = os.environ.get('CUPY_GPU_MEMORY_LIMIT')
    size = None
    fraction = None
    if limit is not None:
        if limit.endswith('%'):
            fraction = float(limit[:-1]) / 100.0
        else:
            size = int(limit)
    return {'size': size, 'fraction': fraction}


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

        # Map from memory pointer of the chunk (intptr_t) to the corresponding
        # Chunk object. All chunks currently allocated to the application from
        # this pool are stored.
        # `_in_use_lock` must be acquired to access it.
        dict _in_use

        # Map from stream identifier to its arena for the stream.
        # `_free_lock` must be acquired to access it.
        dict _arenas

        # Number of total bytes actually allocated on GPU.
        # `_total_bytes_lock` must be acquired to access it.
        size_t _total_bytes

        # Upper limit of the amount to be allocated by this pool.
        # `_total_bytes_lock` must be acquired to access it.
        size_t _total_bytes_limit

        object __weakref__
        object _weakref
        object _free_lock
        object _in_use_lock
        object _total_bytes_lock
        readonly int _device_id

    def __init__(self, allocator=None):
        if allocator is None:
            allocator = _malloc
        self._in_use = {}
        self._arenas = {}
        self._allocator = allocator
        self._weakref = weakref.ref(self)
        self._device_id = device.get_device_id()
        self._free_lock = rlock.create_fastrlock()
        self._in_use_lock = rlock.create_fastrlock()
        self._total_bytes_lock = rlock.create_fastrlock()

        self.set_limit(**(_parse_limit_string()))

    cdef _Arena _arena(self, intptr_t stream_ident):
        """Returns appropriate arena of a given stream.

        All free chunks in the stream belong to one of the bin in the arena.

        Caller is responsible to acquire `_free_lock`.
        """
        ret = self._arenas.get(stream_ident, None)
        if ret is None:
            self._arenas[stream_ident] = ret = _Arena()
        return ret

    cdef MemoryPointer _alloc(self, Py_ssize_t rounded_size):
        if memory_hook._has_memory_hooks():
            hooks = memory_hook.get_memory_hooks()
            if hooks:
                memptr = None
                device_id = self._device_id
                for hook in hooks.values():
                    hook.alloc_preprocess(device_id=device_id,
                                          mem_size=rounded_size)
                try:
                    memptr = self._allocator(rounded_size)
                finally:
                    for hook in hooks.values():
                        mem_ptr = memptr.ptr if memptr is not None else 0
                        hook.alloc_postprocess(device_id=device_id,
                                               mem_size=rounded_size,
                                               mem_ptr=mem_ptr)
                return memptr
        return self._allocator(rounded_size)

    cpdef MemoryPointer malloc(self, size_t size):
        rounded_size = _round_size(size)
        if memory_hook._has_memory_hooks():
            hooks = memory_hook.get_memory_hooks()
            if hooks:
                memptr = None
                device_id = self._device_id
                for hook in hooks.values():
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
                    for hook in hooks.values():
                        hook.malloc_postprocess(device_id=device_id,
                                                size=size,
                                                mem_size=rounded_size,
                                                mem_ptr=mem_ptr,
                                                pmem_id=pmem_id)
                return memptr
        return self._malloc(rounded_size)

    cdef MemoryPointer _malloc(self, size_t size):
        cdef _Chunk chunk
        cdef BaseMemory mem
        cdef PooledMemory pmem
        cdef MemoryPointer ret
        if size == 0:
            return MemoryPointer(Memory(0), 0)

        stream_ident = _get_stream_identifier(
            stream_module.get_current_stream_ptr())

        # find best-fit, or a smallest larger allocation
        gc_mode = _lock_no_gc(self._free_lock)
        try:
            chunk = self._get_chunk(size, stream_ident)
        finally:
            _unlock_no_gc(self._free_lock, gc_mode)

        if chunk is None:
            mem = self._try_malloc(size)
            chunk = _Chunk.__new__(_Chunk)
            # cudaMalloc if a cache is not found
            chunk._init(mem, 0, size, stream_ident)

        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            self._in_use[chunk.ptr()] = chunk
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        pmem = PooledMemory.__new__(PooledMemory)
        pmem._init(chunk, self._weakref)
        ret = MemoryPointer.__new__(MemoryPointer)
        ret._init(pmem, 0)
        return ret

    cpdef free(self, intptr_t ptr, size_t size):
        cdef _Chunk chunk, c

        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            chunk = self._in_use.pop(ptr)
        except KeyError:
            raise RuntimeError('Cannot free out-of-pool memory')
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        stream_ident = chunk.stream_ident

        gc_mode = _lock_no_gc(self._free_lock)
        try:
            arena = self._arena(stream_ident)

            c = chunk.next
            if c is not None and arena.remove_from_free_list(c):
                chunk.merge(c)

            c = chunk.prev
            if c is not None and arena.remove_from_free_list(c):
                c.merge(chunk)
                chunk = c

            arena.append_to_free_list(chunk)
        finally:
            _unlock_no_gc(self._free_lock, gc_mode)

    cpdef free_all_blocks(self, stream=None):
        """Free all **non-split** chunks"""
        cdef intptr_t stream_ident

        with LockAndNoGc(self._free_lock):
            # free blocks in all arenas
            if stream is None:
                for stream_ident in list(self._arenas.iterkeys()):
                    self._compact_index(stream_ident, True)
            else:
                self._compact_index(_get_stream_identifier(stream.ptr), True)

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef size_t n_free_blocks(self):
        cdef size_t n = 0
        cdef _Arena arena
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            for arena in self._arenas.itervalues():
                for v in arena._free:
                    if v is not None:
                        n += len(v)
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return n

    cpdef size_t used_bytes(self):
        cdef size_t size = 0
        cdef _Chunk chunk
        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            for chunk in self._in_use.itervalues():
                size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        return size

    cpdef size_t free_bytes(self):
        cdef size_t size = 0
        cdef set free_list
        cdef _Chunk chunk
        cdef _Arena arena
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            for arena in self._arenas.itervalues():
                for free_list in arena._free:
                    if free_list is None:
                        continue
                    for chunk in free_list:
                        size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return size

    cpdef size_t total_bytes(self):
        with LockAndNoGc(self._total_bytes_lock):
            return self._total_bytes

    cpdef set_limit(self, size=None, fraction=None):
        if size is None:
            if fraction is None:
                size = 0
            else:
                if not 0 <= fraction <= 1:
                    raise ValueError(
                        'memory limit fraction out of range: {}'.format(
                            fraction))
                _, total = runtime.memGetInfo()
                size = fraction * total
            self.set_limit(size=size)
            return

        if fraction is not None:
            raise ValueError('size and fraction cannot be specified at '
                             'one time')
        if size < 0:
            raise ValueError(
                'memory limit size out of range: {}'.format(size))

        with LockAndNoGc(self._total_bytes_lock):
            self._total_bytes_limit = size

    cpdef size_t get_limit(self):
        with LockAndNoGc(self._total_bytes_lock):
            return self._total_bytes_limit

    cdef _compact_index(self, intptr_t stream_ident, bint free):
        # need self._free_lock
        cdef _Arena arena
        cdef list new_free
        cdef set free_list, keep_list
        cdef vector.vector[size_t] new_index
        cdef size_t index
        cdef size_t size_to_free = 0

        if stream_ident not in self._arenas:
            return
        new_free = []
        arena = self._arenas[stream_ident]

        for index, free_list in enumerate(arena._free):
            if not free_list:
                continue
            if free:
                keep_list = set()
                for chunk in free_list:
                    if chunk.prev is not None or chunk.next is not None:
                        keep_list.add(chunk)
                    else:
                        size_to_free += chunk.size
                if len(keep_list) == 0:
                    continue
                free_list = keep_list

            new_index.push_back(arena._index.at(index))
            new_free.append(free_list)
        if free and len(new_free) == 0:
            del self._arenas[stream_ident]
        else:
            arena._free = new_free
            arena._index.swap(new_index)
            arena._flag.assign(new_index.size(), <int8_t>1)
        if size_to_free > 0:
            with LockAndNoGc(self._total_bytes_lock):
                self._total_bytes -= size_to_free

    cdef object _get_chunk(self, size_t size, intptr_t stream_ident):
        # need self._free_lock
        cdef set free_list
        cdef size_t i, index, length
        cdef _Chunk chunk
        cdef size_t bin_index = _bin_index_from_size(size)
        cdef _Arena a = self._arena(stream_ident)
        index = <size_t>(
            algorithm.lower_bound(a._index.begin(), a._index.end(), bin_index)
            - a._index.begin())
        length = a._index.size()
        for i in range(index, length):
            if a._flag.at(i) == 0:
                continue
            free_list = a._free[i]
            chunk = free_list.pop()
            if len(free_list) == 0:
                a._flag[i] = 0
                a._free[i] = None
            if i - index >= _index_compaction_threshold:
                self._compact_index(stream_ident, False)
            remaining = chunk.split(size)
            if remaining is not None:
                a.append_to_free_list(remaining)
            assert chunk.stream_ident == stream_ident
            return chunk
        return None

    cdef BaseMemory _try_malloc(self, size_t size):
        with LockAndNoGc(self._total_bytes_lock):
            total_bytes_limit = self._total_bytes_limit
            total = self._total_bytes + size
            if total_bytes_limit != 0 and total_bytes_limit < total:
                raise OutOfMemoryError(size, total - size, total_bytes_limit)
            self._total_bytes = total

        mem = None
        oom_error = False
        try:
            mem = self._alloc(size).mem
        except CUDARuntimeError as e:
            if e.status != runtime.errorMemoryAllocation:
                raise
            self.free_all_blocks()
            try:
                mem = self._alloc(size).mem
            except CUDARuntimeError as e:
                if e.status != runtime.errorMemoryAllocation:
                    raise
                gc.collect()
                self.free_all_blocks()
                try:
                    mem = self._alloc(size).mem
                except CUDARuntimeError as e:
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    oom_error = True
        finally:
            if mem is None:
                with LockAndNoGc(self._total_bytes_lock):
                    self._total_bytes -= size
                if oom_error:
                    raise OutOfMemoryError(
                        size, total - size, total_bytes_limit)

        return mem


cdef class MemoryPool:

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

    def __init__(self, allocator=None):
        if allocator is None:
            allocator = _malloc
        self._pools = collections.defaultdict(
            lambda: SingleDeviceMemoryPool(allocator))

    cpdef MemoryPointer malloc(self, size_t size):
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
        """Releases free blocks.

        Args:
            stream (cupy.cuda.Stream): Release free blocks in the arena
                of the given stream. The default releases blocks in all
                arenas.

        .. note::
            A memory pool may split a free block for space efficiency. A split
            block is not released until all its parts are merged back into one
            even if :meth:`free_all_blocks` is called.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        mp.free_all_blocks(stream=stream)

    cpdef free_all_free(self):
        """(Deprecated) Use :meth:`free_all_blocks` instead."""
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef size_t n_free_blocks(self):
        """Counts the total number of free blocks.

        Returns:
            int: The total number of free blocks.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.n_free_blocks()

    cpdef size_t used_bytes(self):
        """Gets the total number of bytes used.

        Returns:
            int: The total number of bytes used.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.used_bytes()

    cpdef size_t free_bytes(self):
        """Gets the total number of bytes acquired but not used in the pool.

        Returns:
            int: The total number of bytes acquired but not used in the pool.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.free_bytes()

    cpdef size_t total_bytes(self):
        """Gets the total number of bytes acquired in the pool.

        Returns:
            int: The total number of bytes acquired in the pool.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.total_bytes()

    cpdef set_limit(self, size=None, fraction=None):
        """Sets the upper limit of memory allocation of the current device.

        When `fraction` is specified, its value will become a fraction of the
        amount of GPU memory that is available for allocation.
        For example, if you have a GPU with 2 GiB memory, you can either use
        ``set_limit(fraction=0.5)`` or ``set_limit(size=1024**3)`` to limit
        the memory size to 1 GiB.

        ``size`` and ``fraction`` cannot be specified at one time.
        If both of them are **not** specified or ``0`` is specified, the
        limit will be disabled.

        .. note::
            You can also set the limit by using ``CUPY_GPU_MEMORY_LIMIT``
            environment variable.
            See :ref:`environment` for the details.
            The limit set by this method supersedes the value specified in
            the environment variable.

            Also note that this method only changes the limit for the current
            device, whereas the environment variable sets the default limit for
            all devices.

        Args:
            size (int): Limit size in bytes.
            fraction (float): Fraction in the range of ``[0, 1]``.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        mp.set_limit(size, fraction)

    cpdef size_t get_limit(self):
        """Gets the upper limit of memory allocation of the current device.

        Returns:
            int: The number of bytes
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.get_limit()


cdef class MemoryAsyncPool:
    """(Experimental) CUDA memory pool for all GPU devices on the host.

    A memory pool preserves any allocations even if they are freed by the user.
    One instance of this class can be used for multiple devices. This class
    uses CUDA's Stream Ordered Memory Allocator (supported on CUDA 11.2+).
    The simplest way to use this pool as CuPy's default allocator is the
    following code::

        set_allocator(MemoryAsyncPool().malloc)

    Using this feature requires CUDA >= 11.2 with a supported GPU and platform.
    If it is not supported, an error will be raised.

    The current CuPy stream is used to allocate/free the memory.

    Args:
        pool_handles (str or int): A flag to indicate which mempool to use.
            `'default'` is for the device's default mempool, `'current'` is for
            the current mempool (which could be the default one), and an `int`
            that represents ``cudaMemPool_t`` created from elsewhere for an
            external mempool. A list consisting of these flags can also be
            accepted, in which case the list length must equal to the total
            number of visible devices so that the mempools for each device can
            be set independently.

    .. warning::
        This feature is currently experimental and subject to change.

    .. note::
        :class:`MemoryAsyncPool` currently cannot work with memory hooks.

    .. seealso:: `Stream Ordered Memory Allocator`_

    .. _Stream Ordered Memory Allocator:
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
    """
    # This is an analogous to SingleDeviceMemoryPool + MemoryPool, but for
    # CUDA's async allocator. The main purpose is to provide a memory pool
    # interface for multiple devices, but given that CUDA's mempool is
    # implemented at the driver level, the same pool could be shared by many
    # applications in the same process, so we can't collect meaningful
    # statistics like used bytes for this pool...

    cdef:
        # A list of cudaMemPool_t to each device's mempool
        readonly list _pools

    def __init__(self, pool_handles='default'):
        cdef int dev_id
        if (cpython.PySequence_Check(pool_handles)
                and not isinstance(pool_handles, str)):
            # allow different kinds of handles on each device
            self._pools = [self.set_pool(pool_handles[dev_id], dev_id)
                           for dev_id in range(runtime.getDeviceCount())]
        else:
            # use the same argument for all devices
            self._pools = [self.set_pool(pool_handles, dev_id)
                           for dev_id in range(runtime.getDeviceCount())]

    cdef intptr_t set_pool(self, handle, int dev_id) except? 0:
        cdef intptr_t pool
        if handle == 'default':
            # Use the device's default pool
            pool = runtime.deviceGetDefaultMemPool(dev_id)
        elif handle == 'current':
            # Use the device's current pool
            pool = runtime.deviceGetMemPool(dev_id)
        elif handle == 'create':
            # TODO(leofang): Support cudaMemPoolCreate
            raise NotImplementedError('cudaMemPoolCreate is not yet supported')
        elif isinstance(handle, int):
            # Use an existing pool (likely from other applications?)
            pool = <intptr_t>(handle)
        else:
            raise ValueError("handle must be "
                             "'default' (for the device's default pool), "
                             "'current' (for the device's current pool), "
                             "or int (a pointer to cudaMemPool_t)")
        runtime.deviceSetMemPool(dev_id, pool)
        return pool

    cpdef MemoryPointer malloc(self, size_t size):
        """Allocate memory from the current device's pool on the current
        stream.

        This method can be used as a CuPy memory allocator. The simplest way to
        use a memory pool as the default allocator is the following code::

            set_allocator(MemoryAsyncPool().malloc)

        Args:
            size (int): Size of the memory buffer to allocate in bytes.

        Returns:
            ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

        """
        _util.experimental('cupy.cuda.MemoryAsyncPool.malloc')
        cdef size_t rounded_size = _round_size(size)
        mem = None
        oom_error = False
        try:
            mem = malloc_async(rounded_size)
        except CUDARuntimeError as e:
            if e.status != runtime.errorMemoryAllocation:
                raise
            stream = stream_module.get_current_stream()
            stream.synchronize()
            try:
                mem = malloc_async(rounded_size)
            except CUDARuntimeError as e:
                if e.status != runtime.errorMemoryAllocation:
                    raise
                stream.synchronize()
                try:
                    mem = malloc_async(rounded_size)
                except CUDARuntimeError as e:
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    oom_error = True
        finally:
            if mem is None:
                assert oom_error
                # Set total to -1 as we currently do not keep track of the
                # usage of the async mempool
                raise OutOfMemoryError(size, -1, 0)
        return mem

    cpdef free_all_blocks(self, stream=None):
        # We don't have access to the mempool internal, but if there are
        # any memory asynchronously freed, a synchonization will make sure
        # they become visible (to both cudaMalloc and cudaMallocAsync). See
        # https://github.com/cupy/cupy/issues/3777#issuecomment-758890450
        runtime.deviceSynchronize()

    cpdef size_t n_free_blocks(self):
        raise NotImplementedError

    cpdef size_t used_bytes(self):
        raise NotImplementedError

    cpdef size_t free_bytes(self):
        raise NotImplementedError

    cpdef size_t total_bytes(self):
        raise NotImplementedError

    cpdef set_limit(self, size=None, fraction=None):
        # TODO(leofang): Support cudaMemPoolTrimTo?
        raise NotImplementedError

    cpdef size_t get_limit(self):
        raise NotImplementedError


ctypedef void*(*malloc_func_type)(void*, size_t, int)
ctypedef void(*free_func_type)(void*, void*, int)


cdef intptr_t _call_malloc(
        intptr_t param, intptr_t malloc_func, Py_ssize_t size, int device_id):
    return <intptr_t>(
        (<malloc_func_type>malloc_func)(<void*>param, size, device_id))


cdef void _call_free(
        intptr_t param, intptr_t free_func, intptr_t ptr, int device_id):
    (<free_func_type>free_func)(<void*>param, <void*>ptr, device_id)


@cython.no_gc
cdef class CFunctionAllocatorMemory(BaseMemory):

    def __init__(self, size_t size, intptr_t param,
                 intptr_t malloc_func, intptr_t free_func,
                 int device_id):
        self._param = param
        self._free_func = free_func
        self.device_id = device_id
        self.size = size
        self.ptr = 0
        if size > 0:
            self.ptr = _call_malloc(param, malloc_func, size, device_id)

    def __dealloc__(self):
        if self.ptr:
            _call_free(self._param, self._free_func, self.ptr, self.device_id)


cdef class CFunctionAllocator:

    """Allocator with C function pointers to allocation routines.

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
        param (int): Address of *param*.
        malloc_func (int): Address of *malloc*.
        free_func (int): Address of *free*.
        owner (object): Reference to the owner object to keep the param and
            the functions alive.

    """

    def __init__(self, intptr_t param, intptr_t malloc_func,
                 intptr_t free_func, object owner):
        self._param = param
        self._malloc_func = malloc_func
        self._free_func = free_func
        self._owner = owner

    cpdef MemoryPointer malloc(self, size_t size):
        mem = CFunctionAllocatorMemory(size, self._param, self._malloc_func,
                                       self._free_func, device.get_device_id())
        return MemoryPointer(mem, 0)


cdef class PythonFunctionAllocatorMemory(BaseMemory):

    def __init__(self, size_t size, malloc_func, free_func,
                 int device_id):
        self._free_func = free_func
        self.device_id = device_id
        self.size = size
        self.ptr = 0
        if size > 0:
            self.ptr = malloc_func(size, device_id)

    def __dealloc__(self):
        if self.ptr:
            self._free_func(self.ptr, self.device_id)


cdef class PythonFunctionAllocator:

    """Allocator with python functions to perform memory allocation.

    This allocator keeps functions corresponding to *malloc* and *free*,
    delegating the actual allocation to external sources while only
    handling the timing of the resource allocation and deallocation.

    *malloc* should follow the signature ``malloc(int, int) -> int``
    returning the pointer to the allocated memory given the *param* object,
    the number of bytes to allocate and the device id on which the
    allocation should take place.

    Similarly, *free* should follow the signature
    ``free(int, int)`` with no return, taking the pointer to the
    allocated memory and the device id on which the memory was allocated.

    If the external memory management supports asynchronous operations,
    the current CuPy stream can be retrieved inside ``malloc_func`` and
    ``free_func`` by calling :func:`cupy.cuda.get_current_stream()`. To
    use external streams, wrap them with :func:`cupy.cuda.ExternalStream`.

    Args:
        malloc_func (function): *malloc* function to be called.
        free_func (function): *free* function to be called.

    """

    def __init__(self, malloc_func, free_func):
        self._malloc_func = malloc_func
        self._free_func = free_func

    cpdef MemoryPointer malloc(self, size_t size):
        mem = PythonFunctionAllocatorMemory(
            size, self._malloc_func,
            self._free_func, device.get_device_id())
        return MemoryPointer(mem, 0)

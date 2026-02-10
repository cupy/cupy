# distutils: language = c++
cimport cpython  # NOQA
cimport cython  # NOQA

from cython.operator cimport dereference as deref, postincrement

import atexit
import gc
import os
import threading
import warnings
import weakref

from libc.stdint cimport intptr_t, uintptr_t
from libc.stdint cimport UINT64_MAX
from libc.stdlib cimport malloc as c_malloc
from libc.stdlib cimport free as c_free
from libcpp.atomic cimport atomic as std_atomic
from libcpp.set cimport set as std_set
from libcpp.pair cimport pair as std_pair
from libcpp.mutex cimport mutex as cpp_mutex

from cupy.cuda cimport device
from cupy.cuda cimport memory_hook
from cupy.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime

from cupy_backends.cuda.api.runtime import CUDARuntimeError


cdef extern from "Python.h":
    int PyGC_Enable()
    int PyGC_Disable()
    int PyGC_IsEnabled()


# cudaMalloc() is aligned to at least 512 bytes
# cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
DEF ALLOCATION_UNIT_SIZE = 512
# for test
_allocation_unit_size = ALLOCATION_UNIT_SIZE


cdef bint _exit_mode = False

cdef bint _is_ump_enabled = (int(os.environ.get('CUPY_ENABLE_UMP', '0')) != 0)


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


cdef inline void check_async_alloc_supported(int device_id) except*:
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
        stream (Stream): The stream on which the memory is allocated and freed.
    """

    cdef:
        readonly object stream_ref

    def __init__(self, size_t size, stream):
        self.size = size
        self.device_id = device.get_device_id()
        # The stream is allowed to be destroyed before the memory is freed, so
        # we don't need to hold a strong reference to the stream.
        self.stream_ref = weakref.ref(stream)
        check_async_alloc_supported(self.device_id)
        if size > 0:
            self.ptr = runtime.mallocAsync(size, stream.ptr)

    def __dealloc__(self):
        # Free on the stream on which this memory was allocated.
        # If the stream is already destroyed, free on the current stream. In
        # this case, we trust the user has established a correct stream order.
        if self.ptr == 0:
            return
        stream = self.stream_ref()
        if stream is None:
            stream = stream_module.get_current_stream()
        runtime.freeAsync(self.ptr, stream.ptr)


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
            if device_id == runtime.cudaCpuDeviceId:
                # this happens with SystemMemory...
                device_id = device.get_device_id()
            # CUDA doesn't track memory allocated through the system malloc
            if device_id == runtime.cudaInvalidDeviceId and _is_ump_enabled:
                device_id = device.get_device_id()

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
        if (
            runtime._is_hip_environment and
            driver.get_build_version() < 40300000
        ):
            raise RuntimeError('Managed memory requires ROCm 4.3+')
        self.size = size
        self.device_id = device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = runtime.mallocManaged(size)

    def prefetch(self, stream, *, int device_id=runtime.cudaInvalidDeviceId):
        """(experimental) Prefetch memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream.
            device_id (int): CUDA device ID (-1 for CPU).
        """
        if device_id == runtime.cudaInvalidDeviceId:
            device_id = self.device_id
        runtime.memPrefetchAsync(self.ptr, self.size, device_id, stream.ptr)

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


@cython.no_gc
cdef class SystemMemory(BaseMemory):
    """Memory allocation on an HMM/ATS enabled system.

    HMM stands for heterogeneous memory management. It is a kernel-level
    feature allowing memory allocated via the system ``malloc`` to be
    accessible by both CPU and GPU.

    ATS stands for Address Translation Services. It is a hardware/software
    feature on Grace Hopper that enables the CPU and GPU to share a single
    per-process page table, allowing memory allocated by the system to be
    accessible by both CPU and GPU.

    This class provides an RAII interface of the memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.
    """

    def __init__(self, size_t size):
        self.size = size
        # TODO(leofang): using the GPU id may not be ideal, but setting it
        # to cudaCpuDeviceId (-1) would require a lot of changes
        self.device_id = device.get_device_id()
        self.ptr = 0
        if size > 0:
            self.ptr = <intptr_t>c_malloc(size)
        self._owner = None

    @staticmethod
    cdef from_external(intptr_t ptr, size_t size, object owner):
        """Warp externally allocated (not owned by CuPy) system memory.

        Args:
            ptr (int): Pointer to the buffer.
            size (int): Size of the buffer.
            owner (object): Reference to the owner object to keep the memory
                alive.
        """
        cdef SystemMemory self = SystemMemory.__new__(SystemMemory)
        self.size = size
        # TODO(leofang): using the GPU id may not be ideal, but setting it
        # to cudaCpuDeviceId (-1) would require a lot of changes
        self.device_id = device.get_device_id()
        self.ptr = ptr
        assert owner is not None, 'must provide an owner'
        self._owner = owner

        return self

    def prefetch(self, stream, *, int device_id=runtime.cudaInvalidDeviceId):
        """Prefetch memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream.
            device_id (int): CUDA device ID (-1 for CPU).
        """
        if device_id == runtime.cudaInvalidDeviceId:
            device_id = self.device_id
        runtime.memPrefetchAsync(self.ptr, self.size, device_id, stream.ptr)

    def advise(self, int advise, device.Device dev):
        """Advise about the usage of this memory.

        Args:
            advics (int): Advise to be applied for this memory.
            dev (cupy.cuda.Device): Device to apply the advice for.

        """
        # TODO(leofang): switch to cudaMemAdvice_v2 from CUDA 12.2
        runtime.memAdvise(self.ptr, self.size, advise, dev.id)

    def __dealloc__(self):
        # Note: Cannot raise in the destructor! (cython/cython#1613)
        if self._owner is not None:
            # if we don't own the memory, we must sync before free to avoid
            # any race condition
            runtime.streamSynchronize(stream_module.get_current_stream_ptr())
        elif self.ptr:
            # we don't need to sync because we assume SystemMemory is allocated
            # and protected by the (stream-ordered) memory pool
            c_free(<void*>self.ptr)


@cython.final
@cython.no_gc  # reference cycle would be a bug
cdef class _Chunk:

    """A chunk points to a device memory.

    A chunk might be a splitted memory block from a larger allocation.
    The prev/next pointers construct a doubly-linked list of memory addresses
    sorted by base address that must be contiguous.

    Args:
        mem (~cupy.cuda.Memory): The device memory buffer.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        arena (_Arena): The _Arena this chunk is associated with.

    Attributes:
        mem (Memory): The device memory buffer.
        ptr (int): Memory address.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        arena (_Arena): The _Arena this chunk is associated with (or None
            if this chunk is free'd). Allows us to find the arena on free
            without the need of locking and ensures the arena isn't deleted
            while chunks still exists.
            But, we should set arena=None when we are done with the chunk.

    Notes:
        Mutating chunks is only safe if the arena mutex is held since
        otherwise another thread may mutate it (e.g. split or merge).
    """

    cdef:
        readonly BaseMemory mem
        readonly ptrdiff_t offset
        readonly size_t size
        readonly _Arena arena
        public _Chunk prev
        public _Chunk next

    def __init__(self, *args):
        # For debug
        mem, offset, size, arena = args
        self._init(mem, offset, size, arena)

    def __repr__(self):
        # To simplify debugging if needed.
        mem, offset, prev, next = self.mem, self.offset, self.prev, self.next
        prev = "None" if prev is None else hex(id(prev))
        next = "None" if next is None else hex(id(next))
        return (
            f"<{type(self)} at {hex(id(self))}, {mem=}, {offset=}, "
            f"prev={prev} next={next}>")

    cdef _init(self, BaseMemory mem, ptrdiff_t offset,
               size_t size, _Arena arena):
        assert mem.ptr != 0 or offset == 0
        self.mem = mem
        self.offset = offset
        self.size = size
        self.arena = arena

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
                        self.arena)
        self.size = size

        if self.next is not None:
            remaining.next = self.next
            remaining.next.prev = remaining
        self.next = remaining
        remaining.prev = self
        return remaining

    cpdef merge_next(self):
        """Merge previously splitted next block into this one"""
        self.next.arena = None  # chunk is free so no arena
        self.size += self.next.size
        self.next = self.next.next
        if self.next is not None:
            self.next.prev = self

    # Note on __del__/__dealloc__ of a chunk.
    # If a chunk get's deleted and still has an arena assigned something isn't
    # ideal as we set the arena to `None` when we free chunks explicitly.
    # So this _should_ only happen due to critical errors or at shutdown
    # since if it happens we leak the memory.
    # def __del__(self):
    #    if self.arena is not None:
    #         print("Chunk deleted with active arena, bug?", self)


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
        stream_ptr = stream_module.get_current_stream_ptr()
        if (
            not runtime._is_hip_environment
            and runtime.streamIsCapturing(stream_ptr)
        ):
            raise RuntimeError(
                'the current stream is capturing, so synchronous API calls '
                'are disallowed')
        if size > 0:
            device._enable_peer_access(src.device_id, self.device_id)
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
            device._enable_peer_access(src.device_id, self.device_id)
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
        stream_ptr = stream_module.get_current_stream_ptr()
        if (
            not runtime._is_hip_environment
            and runtime.streamIsCapturing(stream_ptr)
        ):
            raise RuntimeError(
                'the current stream is capturing, so synchronous API calls '
                'are disallowed')
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
        if (
            not runtime._is_hip_environment
            and runtime.streamIsCapturing(stream_ptr)
        ):
            raise RuntimeError(
                'the current stream is capturing, so H2D transfers '
                'are disallowed')
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
        stream_ptr = stream_module.get_current_stream_ptr()
        if (
            not runtime._is_hip_environment
            and runtime.streamIsCapturing(stream_ptr)
        ):
            raise RuntimeError(
                'the current stream is capturing, so synchronous API calls '
                'are disallowed')
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
        if (
            not runtime._is_hip_environment
            and runtime.streamIsCapturing(stream_ptr)
        ):
            raise RuntimeError(
                'the current stream is capturing, so D2H transfers '
                'are disallowed')
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
        stream_ptr = stream_module.get_current_stream_ptr()
        if (
            not runtime._is_hip_environment
            and runtime.streamIsCapturing(stream_ptr)
        ):
            raise RuntimeError(
                'the current stream is capturing, so synchronous API calls '
                'are disallowed')
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


# cpdef because unit-tested
cpdef MemoryPointer _malloc(size_t size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_async(size_t size):
    """Allocate memory from Stream Ordered Memory Allocator.

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

    .. seealso:: `Stream Ordered Memory Allocator`_

    .. _Stream Ordered Memory Allocator:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator
    """
    mem = MemoryAsync(size, stream_module.get_current_stream())
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_managed(size_t size):
    """Allocate managed memory (unified memory).

    This method can be used as a CuPy memory allocator. The simplest way to
    use a managed memory as the default allocator is the following code::

        set_allocator(malloc_managed)

    The advantage using managed memory in CuPy is that device memory
    oversubscription is possible for GPUs that have a non-zero value for the
    device attribute cudaDevAttrConcurrentManagedAccess.
    CUDA >= 8.0 with GPUs later than or equal to Pascal is preferable.

    Read more at: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#axzz4qygc1Ry1  # NOQA

    Args:
        size (int): Size of the memory allocation in bytes.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.
    """
    mem = ManagedMemory(size)
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_system(size_t size):
    """Allocate memory on an HMM/ATS enabled system.

    This method can be used as a CuPy memory allocator. The simplest way to
    use system memory as the default allocator is the following code::

        set_allocator(malloc_system)

    Or, to enable the memory pool support (recommended)::

        set_allocator(MemoryPool(malloc_system).malloc)

    HMM stands for heterogeneous memory management. It is a kernel-level
    feature allowing memory allocated via the system ``malloc`` to be
    accessible by both CPU and GPU. Read more at:
    https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management  # NOQA

    ATS stands for Address Translation Services. It is a hardware/software
    feature on Grace Hopper that enables the CPU and GPU to share a single
    per-process page table, allowing memory allocated by the system to be
    accessible by both CPU and GPU. Read more at:
    https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/  # NOQA

    Args:
        size (int): Size of the memory allocation in bytes.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.
    """
    mem = SystemMemory(size)
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

    cdef intptr_t tid
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
        readonly str identity
        _Chunk chunk
        dict __dict__

    def __init__(self, _Chunk chunk, pool):
        self._init(chunk, pool)

    cdef _init(self, _Chunk chunk, pool):
        self.chunk = chunk
        self.ptr = chunk.ptr()
        self.size = chunk.size
        self.device_id = chunk.mem.device_id
        self.pool = pool

        # we need a way to know what the underlying memory is
        # TODO(leofang): would it be better to do this in MemoryPointer?
        if isinstance(chunk.mem, Memory):
            self.identity = "Memory"
        elif isinstance(chunk.mem, MemoryAsync):
            self.identity = "MemoryAsync"
        elif isinstance(chunk.mem, UnownedMemory):
            self.identity = "UnownedMemory"
        elif isinstance(chunk.mem, SystemMemory):
            self.identity = "SystemMemory"
            self.prefetch = <SystemMemory>(chunk.mem).prefetch
            self.advise = <SystemMemory>(chunk.mem).advise
        elif isinstance(chunk.mem, ManagedMemory):
            self.identity = "ManagedMemory"
            self.prefetch = <ManagedMemory>(chunk.mem).prefetch
            self.advise = <ManagedMemory>(chunk.mem).advise
        elif isinstance(chunk.mem, CFunctionAllocatorMemory):
            self.identity = "CFunctionAllocatorMemory"
        elif isinstance(chunk.mem, PythonFunctionAllocatorMemory):
            self.identity = "PythonFunctionAllocatorMemory"
        else:
            self.identity = "<unknown>"

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
                    (<SingleDeviceMemoryPool>pool).free(self.chunk)
                finally:
                    for hook in hooks.values():
                        hook.free_postprocess(device_id=device_id,
                                              mem_size=size,
                                              mem_ptr=ptr,
                                              pmem_id=pmem_id)
                return
        (<SingleDeviceMemoryPool>pool).free(self.chunk)

    def __dealloc__(self):
        if _exit_mode:
            return  # To avoid error at exit
        self.free()


cpdef inline size_t _round_size(size_t size):
    """Rounds up the memory size to fit memory alignment of cudaMalloc."""
    # avoid 0 div checking
    size = (size + ALLOCATION_UNIT_SIZE - 1) // ALLOCATION_UNIT_SIZE
    return size * ALLOCATION_UNIT_SIZE


# The std::set contains the size, _Chunk pair (as uintptr_t)
# The uintptr_t means we can compare well defined (including to 0)
ctypedef std_pair[size_t, uintptr_t] index_type


@cython.final
cdef class _Arena:
    # Arena class managing all free chunks belonging to a single stream ident.
    # A few notes on safety:
    #   * All access to `index` requires a lock, the callers must ensure this.
    #     these means almost all methods must be called with a lock held.
    #   * `_add_pending_free_atomic` is atomic and safe without a lock, though.
    #   * No custom `__dealloc__`: the lifetime is tied to the `_Chunks`.
    #     Only an empty Arena can be free'd (except maybe at shutdown).
    #
    # The index owns all chunks, we have one entry per chunk (rather than
    # buckets) for simplicity. Making it a `map` to buckets may be useful
    # including possibly to try and achieve less locking.
    cdef:
        list _pending_free  # "lock free" list to stage chunks
        std_set[index_type] index  # bin_size, _Chunk
        cdef object __weakref__

    def __cinit__(self):
        self._pending_free = []

    cdef add_pending_free_atomic(self, _Chunk chunk):
        """Add a chunk back to the arena as free. If you are holding the lock,
        use `insert_chunk()` directly (it will merge unless `merge=False`).
        """
        # In theory we could try to lock here, but in practice the caller
        # should handle that when relevant.
        self._pending_free.append(chunk)

    cdef _commit_pending_free(self):
        """Clean up the current free list. An exclusive lock must
        be held for this, since we may have to insert.

        This also attempts to merge chunks that have been split up.
        """
        cdef Py_ssize_t i
        cdef _Chunk chunk

        for i in range(len(self._pending_free)):
            chunk = self._pending_free.pop()
            self.insert_chunk(chunk)

    cdef _Chunk try_merge_chunk(self, _Chunk chunk):
        # If this chunk was split try to merge it again.
        while chunk.next is not None and self.try_remove_chunk(chunk.next):
            chunk.merge_next()

        while chunk.prev is not None and self.try_remove_chunk(chunk.prev):
            chunk = chunk.prev
            chunk.merge_next()

        return chunk

    cdef insert_chunk(self, _Chunk chunk, bint merge=True):
        if merge:
            chunk = self.try_merge_chunk(chunk)

        self.index.insert(index_type(chunk.size, <uintptr_t><void *>chunk))
        cpython.Py_INCREF(chunk)  # index holds a reference now

    cdef size_t free_all(self) except -1:
        """Frees all chunks (that can be free'd, split ones cannot).
        """
        cdef _Chunk chunk
        cdef size_t bytes_freed = 0

        self._commit_pending_free()

        it = self.index.begin()
        while it != self.index.end():
            chunk = <_Chunk><void *>deref(it).second
            if chunk.next is not None or chunk.prev is not None:
                # Cannot free this chunk, continue to next.
                postincrement(it)
                continue

            # Otherwise, advance iterator and remove chunk
            self.index.erase(postincrement(it))
            cpython.Py_DECREF(chunk)

            # Freeing means removing the chunks from this arena. Because
            # chunks hold on the arena, we need to break that cycle.
            chunk.arena = None
            bytes_freed += chunk.size
            del chunk

        return bytes_freed

    cdef bint try_remove_chunk(self, _Chunk chunk) except -1:
        """Checks if the chunk appears to be currently free and if yes
        unlinks (removes) it from the arena (free chunks).
        This is used for merging previously split chunks again.

        Returns:
            bool: ``True`` if the chunk can successfully be removed from
            the free list. ``False`` if the chunk is not free.
        """
        cdef std_set[index_type].iterator it

        it = self.index.find(index_type(chunk.size, <uintptr_t><void *>chunk))
        if it == self.index.end():
            return False  # chunk seems not to be free

        chunk = <_Chunk><void *>deref(it).second
        self.index.erase(it)  # Remove and decref
        cpython.Py_DECREF(chunk)
        return True

    cdef _Chunk get_chunk(self, size_t size):
        """Get a free chunk of at least the given size from the arena.
        """
        cdef std_set[index_type].iterator it
        cdef _Chunk chunk = None

        it = self.index.lower_bound(index_type(size, 0))
        if it != self.index.end():
            chunk = <_Chunk><void *>deref(it).second
            self.index.erase(it)  # Remove and decref
            cpython.Py_DECREF(chunk)
        else:
            return None

        remaining = chunk.split(size)
        if remaining is not None:
            self.insert_chunk(remaining, merge=False)
        return chunk

    cdef _get_size(self):
        return self.index.size()

    cdef _index_to_python(self):
        # For debug purpose, expose index into Python.
        cdef std_set[index_type].iterator it
        cdef list result = []

        self._commit_pending_free()

        it = self.index.begin()
        while it != self.index.end():
            result.append((deref(it).first, <object><void *>deref(it).second))
            postincrement(it)

        return result


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

        # Arenas are stored as weak references inside this dict (very minimal
        # WeakValueDict). They must only be taken via `_arena(ident)` or the
        # `_Chunk` attribute that keeps the arena alive.
        # NOTE: If you work with any arena you must hold the `_arena_mutex`.
        # The only safe operation is `add_pending_free_atomic` (note the
        # atomic # in the name).
        dict _arenas
        # NOTE: Never use `.lock()` outside a `with nogil:` statement as it
        # may deadlock (GIL may be unlocked and another thread also locks).
        cpp_mutex _arena_mutex

        # Number of total bytes actually allocated on GPU.
        # NOTE: You MUST use _try_block_total_bytes to increase the value
        # (This ensures correct checks for whether memory is available).
        std_atomic[size_t] _total_bytes
        # Number of used bytes, modify with normal +=/-=.
        std_atomic[size_t] _in_use_bytes
        # Upper limit of the amount to be allocated by this pool, we don't
        # care too much about thread-safety for it, but make it atomic anyway.
        std_atomic[size_t] _total_bytes_limit

        object __weakref__
        object _weakref
        readonly int _device_id

    def __init__(self, allocator=None):
        if allocator is None:
            allocator = _malloc
        self._arenas = {}
        self._allocator = allocator
        self._weakref = weakref.ref(self)
        self._device_id = device.get_device_id()

        self.set_limit(**(_parse_limit_string()))

    cpdef _Arena _arena(self, intptr_t stream_ident):
        """Returns appropriate arena of a given stream, you should hold the
        arena mutex when getting this (to ensure nobody else changes things)

        All free chunks in the stream belong to one of the bin in the arena.
        """
        cdef cpython.PyObject *ret
        ref = self._arenas.get(stream_ident, None)
        if ref is not None:
            if cpython.PyWeakref_GetRef(ref, &ret) == 1:
                return <object>ret

        # Assume there are not many arenas being created and deleted so
        # just clean up dead refs here.
        for key in list(self._arenas):
            if self._arenas[key]() is None:
                del self._arenas[key]

        new_arena = _Arena()
        ref = weakref.ref(new_arena)
        self._arenas[stream_ident] = ref
        return new_arena

    def _debug_arena_get_index(self, intptr_t stream_ident):
        # Sets returned are not copies, mutating them will break things
        with nogil:
            self._arena_mutex.lock()
        try:
            return self._arena(stream_ident)._index_to_python()
        finally:
            self._arena_mutex.unlock()

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
        cdef _Arena arena
        cdef _Chunk chunk
        cdef BaseMemory mem
        cdef PooledMemory pmem
        cdef MemoryPointer ret
        if size == 0:
            return MemoryPointer(Memory(0), 0)

        stream_ident = _get_stream_identifier(
            stream_module.get_current_stream_ptr())

        if not self._arena_mutex.try_lock():
            with nogil:
                self._arena_mutex.lock()
        try:
            arena = self._arena(stream_ident)
            # find best-fit, or a smallest larger allocation
            chunk = arena.get_chunk(size)
        finally:
            self._arena_mutex.unlock()

        if chunk is None:
            # cudaMalloc if a cache chunk is not found
            mem = self._try_malloc(size)
            chunk = _Chunk.__new__(_Chunk)
            chunk._init(mem, 0, size, arena)

        self._in_use_bytes += chunk.size

        pmem = PooledMemory.__new__(PooledMemory)
        pmem._init(chunk, self._weakref)
        ret = MemoryPointer.__new__(MemoryPointer)
        ret._init(pmem, 0)
        return ret

    cdef free(self, _Chunk chunk):
        self._in_use_bytes -= chunk.size

        # Make sure freeing is always safe, but if we can lock do it.
        if self._arena_mutex.try_lock():
            try:
                chunk.arena.insert_chunk(chunk)
            finally:
                self._arena_mutex.unlock()
        else:
            chunk.arena.add_pending_free_atomic(chunk)

    cpdef free_all_blocks(self, stream=None):
        """Free all **non-split** blocks for one or all arenas.
        """
        cdef intptr_t stream_ident
        cdef _Arena arena
        cdef tuple idents
        cdef size_t bytes_freed = 0

        if not self._arena_mutex.try_lock():
            with nogil:
                self._arena_mutex.lock()
        try:
            if stream is None:
                idents = tuple(self._arenas.keys())
            else:
                stream_ident = _get_stream_identifier(stream.ptr)
                if stream_ident not in self._arenas:
                    return  # stream doesn't exist, just return
                idents = (stream_ident,)

            for ident in idents:
                arena = self._arenas[ident]()
                if arena is None:
                    del self._arenas[ident]
                else:
                    bytes_freed += arena.free_all()
        finally:
            self._arena_mutex.unlock()
            self._total_bytes -= bytes_freed

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef size_t n_free_blocks(self) except -1:
        cdef size_t n = 0
        cdef _Arena arena

        if not self._arena_mutex.try_lock():
            with nogil:
                self._arena_mutex.lock()
        try:
            for ref in self._arenas.itervalues():
                arena = ref()
                if arena is None:
                    continue
                n += arena._get_size()
        finally:
            self._arena_mutex.unlock()

        return n

    cpdef size_t used_bytes(self):
        """Currently used bytes."""
        return self._in_use_bytes.load()

    cpdef size_t free_bytes(self):
        return self._total_bytes.load() - self._in_use_bytes.load()

    cpdef size_t total_bytes(self):
        return self._total_bytes.load()

    cdef bint _try_block_total_bytes(self, size_t size) except -1:
        """Try to block off `size` bytes from the total pool size.
        Returns True if successfull (caller should try to allocate that many
        bytes) and False if allocation would go above the threshold.
        May raise `OutOfMemoryError` if `size` is too large for the pool.
        """
        cdef size_t limit = self._total_bytes_limit.load()
        cdef size_t curr_total_bytes = self._total_bytes.load()
        cdef bint limit_ok

        if limit == 0:
            limit = UINT64_MAX

        if size > limit:
            # Check also protects against integer overflow below.
            raise OutOfMemoryError(size, self._total_bytes.load(), limit)

        limit_ok = curr_total_bytes <= limit - size

        while limit_ok and not self._total_bytes.compare_exchange_weak(
                curr_total_bytes, curr_total_bytes+size):
            # If we reach here, `_total_bytes` was changed by another thread.
            # TODO(seberg): Presumably we could use weaker memory order.
            curr_total_bytes = self._total_bytes.load()
            limit_ok = curr_total_bytes <= limit - size

        return limit_ok

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

        self._total_bytes_limit.store(size)

    cpdef size_t get_limit(self):
        return self._total_bytes_limit.load()

    cdef BaseMemory _try_malloc(self, size_t size):
        cdef size_t limit = self._total_bytes_limit.load()
        cdef bint ok

        if not (ok := self._try_block_total_bytes(size)):
            self.free_all_blocks(None)
        if not ok and not (ok := self._try_block_total_bytes(size)):
            gc.collect()
            self.free_all_blocks(None)
        if not ok and not (ok := self._try_block_total_bytes(size)):
            raise OutOfMemoryError(size, self._total_bytes.load(), limit)

        # If we reach here, we are allowed to allocate size bytes based on
        # the limit. But the actual allocation may still fail (and we must
        # reduce _total_bytes again if it does).
        mem = None
        oom_error = False
        try:
            mem = self._alloc(size).mem
        except CUDARuntimeError as e:
            if e.status != runtime.errorMemoryAllocation:
                raise

            self.free_all_blocks(None)
            try:
                mem = self._alloc(size).mem
            except CUDARuntimeError as e:
                if e.status != runtime.errorMemoryAllocation:
                    raise
                gc.collect()
                gc.collect()
                self.free_all_blocks(None)
                try:
                    mem = self._alloc(size).mem
                except CUDARuntimeError as e:
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    oom_error = True
        finally:
            if mem is None:
                self._total_bytes -= size
                if oom_error:
                    raise OutOfMemoryError(
                        size, self._total_bytes.load(), limit)

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
        self._allocator = allocator

    cdef _ensure_pools_and_return_device_pool(self):
        # assume we get to create the pools (we may not be the only one)
        n_gpu = runtime.getDeviceCount()
        pools = tuple(
            SingleDeviceMemoryPool(self._allocator) for i in range(n_gpu))

        # If no-one beat us to it, set _pools. Note that the above seems to
        # release the critical section, so including it doesn't work and
        # would require a proper mutex.
        with cython.critical_section(self):
            if self._pools is None:
                self._pools = pools

        return self._pools[device.get_device_id()]

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
        mp = <SingleDeviceMemoryPool>self.device_pool()
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
        mp = <SingleDeviceMemoryPool>self.device_pool()
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
        mp = <SingleDeviceMemoryPool>self.device_pool()
        return mp.n_free_blocks()

    cpdef size_t used_bytes(self):
        """Gets the total number of bytes used by the pool.

        Returns:
            int: The total number of bytes used by the pool.
        """
        mp = <SingleDeviceMemoryPool>self.device_pool()
        return mp.used_bytes()

    cpdef size_t free_bytes(self):
        """Gets the total number of bytes acquired but not used by the pool.

        Returns:
            int: The total number of bytes acquired but not used by the pool.
        """
        mp = <SingleDeviceMemoryPool>self.device_pool()
        return mp.free_bytes()

    cpdef size_t total_bytes(self):
        """Gets the total number of bytes acquired by the pool.

        Returns:
            int: The total number of bytes acquired by the pool.
        """
        mp = <SingleDeviceMemoryPool>self.device_pool()
        return mp.total_bytes()

    cpdef set_limit(self, size=None, fraction=None):
        """Sets the upper limit of memory allocation of the current device.

        When `fraction` is specified, its value will become a fraction of the
        amount of GPU memory that is available for allocation.
        For example, if you have a GPU with 2 GiB memory, you can either use
        ``set_limit(fraction=0.5)`` or ``set_limit(size=1024**3)`` to limit
        the memory size to 1 GiB.

        ``size`` and ``fraction`` cannot be specified at the same time.
        If both of them are **not** specified or ``0`` is specified, the
        limit will be disabled.

        .. note::
            You can also set the limit by using ``CUPY_GPU_MEMORY_LIMIT``
            environment variable, see :ref:`environment` for the details.
            The limit set by this method supersedes the value specified in
            the environment variable.

            Also note that this method only changes the limit for the current
            device, whereas the environment variable sets the default limit for
            all devices.

        .. note::
            Changing the limit is not thread-safe. Other threads may use an
            outdated value for example if currently performing allocations.

        Args:
            size (int): Limit size in bytes.
            fraction (float): Fraction in the range of ``[0, 1]``.
        """
        mp = <SingleDeviceMemoryPool>self.device_pool()
        mp.set_limit(size, fraction)

    cpdef size_t get_limit(self):
        """Gets the upper limit of memory allocation of the current device.

        Returns:
            int: The number of bytes
        """
        mp = <SingleDeviceMemoryPool>self.device_pool()
        return mp.get_limit()


cdef class MemoryAsyncPool:
    """CUDA memory pool for all GPU devices on the host.

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

    .. note::
        :class:`MemoryAsyncPool` currently cannot work with memory hooks.

    .. seealso:: `Stream Ordered Memory Allocator`_

    .. _Stream Ordered Memory Allocator:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator
    """
    # This is an analogous to SingleDeviceMemoryPool + MemoryPool, but for
    # CUDA's async allocator. The main purpose is to provide a memory pool
    # interface for multiple devices. Given that CUDA's mempool is implemented
    # at the driver level, the same pool can be shared by many applications
    # in the same process.
    #
    # Internally (as of driver v11.3) the pool starts with size 0. The first
    # allocation will bump the size to 32n MiB to accommodate the requested
    # amount (n is integer). The size is increased only if it is not enough
    # to meet later allocation needs. The size is decreased to 32m MiB (m<n)
    # if enough memory is returned (freed) to the pool when free_all_blocks()
    # (that is, sync + cudaMemPoolTrimTo) is called. This observation may
    # vary with future driver updates, so the MemoryAsyncPool API does not
    # rely on any internal behavior, but only on the Programming Guide and
    # sane assumptions.

    cdef:
        # A list of cudaMemPool_t to each device's mempool
        readonly list _pools
        readonly bint memoryAsyncHasStat

    def __init__(self, pool_handles='current'):
        cdef int dev_id, prev_dev_id, dev_counts
        cdef dict limit = _parse_limit_string()
        dev_counts = runtime.getDeviceCount()
        self._pools = []
        self.memoryAsyncHasStat = (runtime.driverGetVersion() >= 11030)
        prev_dev_id = runtime.getDevice()
        if (cpython.PySequence_Check(pool_handles)
                and not isinstance(pool_handles, str)):
            # allow different kinds of handles on each device
            for dev_id in range(dev_counts):
                try:
                    runtime.setDevice(dev_id)
                    self._pools.append(self.set_pool(
                        pool_handles[dev_id], dev_id))
                    self.set_limit(**limit)
                finally:
                    runtime.setDevice(prev_dev_id)
        else:
            # use the same argument for all devices
            for dev_id in range(dev_counts):
                try:
                    runtime.setDevice(dev_id)
                    self._pools.append(self.set_pool(pool_handles, dev_id))
                    self.set_limit(**limit)
                finally:
                    runtime.setDevice(prev_dev_id)

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
        cdef size_t rounded_size = _round_size(size)
        mem = None
        oom_error = False

        # CUDA does not allow us to set a hard limit, so the best we can do is
        # to prevent CuPy from drawing too much memory from the pool; we cannot
        # do anything if other applications oversubscribe the pool.
        cdef size_t curr_total=-1, curr_free=0, total_limit=0
        if self.memoryAsyncHasStat:
            curr_total = self.total_bytes()
            curr_free = curr_total - self.used_bytes()
            if curr_free < rounded_size:  # need to increase pool size
                total_limit = self.get_limit()
                if max(total_limit, curr_total) < curr_total + rounded_size:
                    raise OutOfMemoryError(size, curr_total, total_limit)

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
                # Set total to -1 as we do not have access to the mempool usage
                raise OutOfMemoryError(size, curr_total, total_limit)
        return mem

    cpdef free_all_blocks(self, stream=None):
        """Releases free memory.

        Args:
            stream (cupy.cuda.Stream): Release memory freed on the given
                ``stream``. If ``stream`` is ``None``, the current stream is
                used.

        .. seealso:: `Physical Page Caching Behavior`_

        .. _Physical Page Caching Behavior:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-physical-page-caching-behavior
        """
        # We don't have access to the mempool internal, but if there are
        # any memory asynchronously freed, a synchronization will make sure
        # they become visible (to both cudaMalloc and cudaMallocAsync). See
        # https://github.com/cupy/cupy/issues/3777#issuecomment-758890450
        if stream is None:
            stream = stream_module.get_current_stream()
        stream.synchronize()
        cdef intptr_t pool = self._pools[device.get_device_id()]
        # We don't care the actual limit; putting 0 here means we guarantee
        # to reserve at least 0 bytes
        runtime.memPoolTrimTo(pool, 0)

    cpdef size_t n_free_blocks(self):
        raise NotImplementedError(
            'This function is not supported in MemoryAsyncPool')

    cpdef size_t used_bytes(self) except*:
        """Gets the total number of bytes used by the pool.

        Returns:
            int: The total number of bytes used by the pool.
        """
        if not self.memoryAsyncHasStat:
            raise RuntimeError(
                'The driver version is insufficient for this query')
        cdef intptr_t pool = self._pools[device.get_device_id()]
        return runtime.memPoolGetAttribute(
            pool, runtime.cudaMemPoolAttrUsedMemCurrent)

    cpdef size_t free_bytes(self) except*:
        """Gets the total number of bytes acquired but not used by the pool.

        Returns:
            int: The total number of bytes acquired but not used by the pool.
        """
        return self.total_bytes() - self.used_bytes()

    cpdef size_t total_bytes(self) except*:
        """Gets the total number of bytes acquired by the pool.

        Returns:
            int: The total number of bytes acquired by the pool.
        """
        if not self.memoryAsyncHasStat:
            raise RuntimeError(
                'The driver version is insufficient for this query')
        cdef intptr_t pool = self._pools[device.get_device_id()]
        return runtime.memPoolGetAttribute(
            pool, runtime.cudaMemPoolAttrReservedMemCurrent)

    cpdef set_limit(self, size=None, fraction=None):
        """Sets the upper limit of memory allocation of the current device.

        When `fraction` is specified, its value will become a fraction of the
        amount of GPU memory that is available for allocation.
        For example, if you have a GPU with 2 GiB memory, you can either use
        ``set_limit(fraction=0.5)`` or ``set_limit(size=1024**3)`` to limit
        the memory size to 1 GiB.

        ``size`` and ``fraction`` cannot be specified at the same time.
        If both of them are **not** specified or ``0`` is specified, the
        limit will be disabled.

        .. note::
            Unlike with :class:`MemoryPool`, :class:`MemoryAsyncPool`'s
            :meth:`set_limit` method can only impose a *soft* limit. If other
            (non-CuPy) applications are also allocating memory from the same
            mempool, this limit may not be respected. Internally, this limit
            is set via the ``cudaMemPoolAttrReleaseThreshold`` attribute.

        .. note::
            You can also set the limit by using ``CUPY_GPU_MEMORY_LIMIT``
            environment variable, see :ref:`environment` for the details.
            The limit set by this method supersedes the value specified in
            the environment variable.

            Also note that this method only changes the limit for the current
            device, whereas the environment variable sets the default limit for
            all devices.

        Args:
            size (int): Limit size in bytes.
            fraction (float): Fraction in the range of ``[0, 1]``.
        """
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

        if size == 0:
            size = UINT64_MAX  # ensure pool size is never shrunk
        cdef intptr_t pool = self._pools[device.get_device_id()]
        runtime.memPoolSetAttribute(
            pool, runtime.cudaMemPoolAttrReleaseThreshold, size)

    cpdef size_t get_limit(self):
        """Gets the upper limit of memory allocation of the current device.

        Returns:
            int: The number of bytes

        .. note::
            Unlike with :class:`MemoryPool`, :class:`MemoryAsyncPool`'s
            :meth:`set_limit` method can only impose a *soft* limit. If other
            (non-CuPy) applications are also allocating memory from the same
            mempool, this limit may not be respected.
        """
        cdef intptr_t pool = self._pools[device.get_device_id()]
        return runtime.memPoolGetAttribute(
            pool, runtime.cudaMemPoolAttrReleaseThreshold)


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

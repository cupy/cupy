import collections
import ctypes
import weakref

from cupy.cuda import device
from cupy.cuda import runtime


class Memory(object):

    """Memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        size (int): Size of the memory allocation in bytes.

    """
    def __init__(self, size):
        self.size = size
        self._device = None
        self.ptr = ctypes.c_void_p()
        if size > 0:
            self._device = device.Device()
            self.ptr = runtime.malloc(size)

    def __del__(self):
        if self.ptr:
            runtime.free(self.ptr)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        return self.ptr.value or 0

    @property
    def device(self):
        """Device whose memory the pointer refers to."""
        return self._device


class MemoryPointer(object):

    """Pointer to a point on a device memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (Memory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        mem (Memory): The device memory buffer.
        ptr (ctypes.c_void_p): Pointer to the place within the buffer.

    """
    def __init__(self, mem, offset):
        self.mem = mem
        self._device = mem.device
        self.ptr = ctypes.c_void_p(int(mem) + int(offset))

    def __int__(self):
        """Returns the pointer value."""
        return self.ptr.value or 0

    def __add__(self, offset):
        """Adds an offset to the pointer."""
        return MemoryPointer(self.mem, int(self) - int(self.mem) + offset)

    def __radd__(self, offset):
        """Adds an offset to the pointer."""
        return self + offset

    def __iadd__(self, offset):
        """Adds an offset to the pointer in place."""
        self.ptr.value += offset
        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    @property
    def device(self):
        """Device whose memory the pointer refers to."""
        return self._device

    def copy_from_device(self, src, size):
        """Copies a memory sequence from a (possibly different) device.

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpy(self.ptr, src.ptr, size,
                           runtime.memcpyDeviceToDevice)

    def copy_from_device_async(self, src, size, stream):
        """Copies a memory sequence from a (possibly different) device asynchronously.

        Args:
            src (cupy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            runtime.memcpyAsync(self.ptr, src.ptr, size,
                                runtime.memcpyDeviceToDevice, stream)

    def copy_from_host(self, mem, size):
        """Copies a memory sequence from the host memory.

        Args:
            mem (ctypes.c_void_p): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpy(self.ptr, mem, size, runtime.memcpyHostToDevice)

    def copy_from_host_async(self, mem, size, stream):
        """Copies a memory sequence from the host memory asynchronously.

        Args:
            src (ctypes.c_void_p): Source memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpyAsync(self.ptr, mem, size, stream,
                                runtime.memcpyHostToDevice)

    def copy_from(self, mem, size):
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

    def copy_from_async(self, mem, size, stream):
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

    def copy_to_host(self, mem, size):
        """Copies a memory sequence to the host memory.

        Args:
            mem (ctypes.c_void_p): Target memory pointer.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memcpy(mem, self.ptr, size, runtime.memcpyDeviceToHost)

    def copy_to_host_async(self, mem, size, stream):
        """Copies a memory sequence to the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Target memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            runtime.memcpyAsync(mem, self.ptr, size,
                                runtime.memcpyDeviceToHost, stream)

    def memset(self, value, size):
        """Fills a memory sequence by constant byte value.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            runtime.memset(self.ptr, value, size)

    def memset_async(self, value, size, stream):
        """Fills a memory sequence by constant byte value asynchronously.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            runtime.memsetAsync(self.ptr, value, size, stream)


def _malloc(size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


_current_allocator = _malloc


def alloc(size):
    """Calls the current allocator.

    Use :func:`~cupy.cuda.set_allocator` to change the current allocator.

    Args:
        size (int): Size of the memory allocation.

    Returns:
        ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

    """
    return _current_allocator(size)


def set_allocator(allocator=_malloc):
    """Sets the current allocator.

    Args:
        allocator (function): CuPy memory allocator. It must have the same
            interface as the :func:`cupy.cuda.alloc` function, which takes the
            buffer size as an argument and returns the device buffer of that
            size.

    """
    global _current_allocator
    _current_allocator = allocator


class PooledMemory(Memory):

    """Memory allocation for a memory pool.

    The instance of this class is created by memory pool allocator, so user
    should not instantiate it by hand.

    """
    def __init__(self, mem, pool):
        self.ptr = mem.ptr
        self.size = mem.size
        self._device = mem._device
        self.pool = weakref.ref(pool)

    def __del__(self):
        if self.ptr is not None:
            self.free()

    def free(self):
        """Frees the memory buffer and returns it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        pool = self.pool()
        if pool:
            pool.free(self.ptr, self.size)
        self.ptr = None
        self.size = 0
        self._device = None
        self.pool = None


class SingleDeviceMemoryPool(object):

    """Memory pool implementation for single device."""

    def __init__(self, allocator=_malloc):
        self._in_use = {}
        self._free = collections.defaultdict(list)
        self._alloc = allocator

    def malloc(self, size):
        free = self._free[size]

        if free:
            mem = free.pop()
        else:
            try:
                mem = self._alloc(size).mem
            except runtime.CUDARuntimeError as e:
                if e.status != 2:
                    raise
                self.free_all_free()
                mem = self._alloc(size).mem

        self._in_use[mem.ptr.value] = mem
        pmem = PooledMemory(mem, self)
        return MemoryPointer(pmem, 0)

    def free(self, ptr, size):
        mem = self._in_use.pop(ptr.value, None)
        if mem is None:
            raise RuntimeError('Cannot free out-of-pool memory')
        else:
            self._free[size].append(mem)

    def free_all_free(self):
        self._free = collections.defaultdict(list)


class MemoryPool(object):

    """Memory pool for all devices on the machine.

    A memory pool preserves any allocations even if they are freed by the user.
    Freed memory buffers are held by the memory pool as *free blocks*, and they
    are reused for further memory allocations of the same sizes. The allocated
    blocks are managed for each device, so one instance of this class can be
    used for multiple devices.

    .. note::
       When the allocation is skipped by reusing the pre-allocated block, it
       does not call cudaMalloc and therefore CPU-GPU synchronization does not
       occur. It makes interleaves of memory allocations and kernel invocations
       very fast.

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

    def malloc(self, size):
        """Allocates the memory, from the pool if possible.

        This method can be used as a CuPy memory allocator. The simplest way to
        use a memory pool as the default allocator is the following code::

           set_allocator(MemoryPool().malloc)

        Args:
            size (int): Size of the memory buffer to allocate in bytes.

        Returns:
            ~cupy.cuda.MemoryPointer: Pointer to the allocated buffer.

        """
        dev = device.Device().id
        return self._pools[dev].malloc(size)

import collections
import ctypes
import weakref

from cupy.cuda import device
from cupy.cuda import driver
from cupy.cuda import runtime


class Memory(object):

    def __init__(self, size):
        if size > 0:
            self.ptr = runtime.malloc(size)
        else:
            self.ptr = ctypes.c_void_p()
        self.size = size

    def __del__(self):
        runtime.free(self.ptr)

    def __int__(self):
        return self.ptr.value or 0


class MemoryPointer(object):

    def __init__(self, mem, offset):
        self.mem = mem
        self.ptr = ctypes.c_void_p(int(mem) + offset)

    def __int__(self):
        return self.ptr.value or 0

    def __add__(self, offset):
        return MemoryPointer(self.mem, int(self) - int(self.mem) + offset)

    def __radd__(self, offset):
        return self + offset

    def __iadd__(self, offset):
        self.ptr.value += offset
        return self

    def __sub__(self, offset):
        return self + -offset

    def __isub__(self, offset):
        return self.__iadd__(-offset)

    @property
    def device(self):
        return device.from_pointer(self.ptr)

    def copy_from_device(self, src, size):
        if size > 0:
            runtime.memcpyDtoD(self.ptr, src.ptr, size)

    def copy_from_device_async(self, src, size, stream):
        if size > 0:
            runtime.memcpyDtoDAsync(self.ptr, src.ptr, size, stream)

    def copy_from_host(self, mem, size):
        if size > 0:
            runtime.memcpyHtoD(self.ptr, mem, size)

    def copy_from_host_async(self, mem, size, stream):
        if size > 0:
            runtime.memcpyHtoDAsync(self.ptr, mem, size, stream)

    def copy_from_peer(self, src, size):
        if size > 0:
            runtime.memcpyPeer(self.ptr, self.device, src.ptr, src.device,
                               size)

    def copy_from_peer_async(self, src, size, stream):
        if size > 0:
            runtime.memcpyPeerAsync(self.ptr, self.device, src.ptr, src.device,
                                    size, stream)

    def copy_from(self, mem, size):
        if isinstance(mem, MemoryPointer):
            if self.device == mem.device:
                self.copy_from_device(mem, size)
            else:
                self.copy_from_peer(mem, size)
        else:
            self.copy_from_host(mem, size)

    def copy_from_async(self, mem, size, stream):
        if isinstance(mem, MemoryPointer):
            if self.device == mem.device:
                self.copy_from_device_async(mem, size, stream)
            else:
                self.copy_from_peer_async(mem, size, stream)
        else:
            self.copy_from_host_async(mem, size, stream)

    def copy_to_host(self, mem, size):
        if size > 0:
            runtime.memcpyDtoH(mem, self.ptr, size)

    def copy_to_host_async(self, mem, size, stream):
        if size > 0:
            runtime.memcpyDtoHAsync(mem, self.ptr, size, stream)

    def memset(self, value, size):
        if size > 0:
            runtime.memset(self.ptr, value, size)

    def memset_async(self, value, size, stream):
        if size > 0:
            runtime.memsetAsync(self.ptr, value, size, stream)

    def memset32(self, value, size):
        if size > 0:
            if isinstance(value, float):
                value = ctypes.cast((ctypes.c_float * 1)(value),
                                    ctypes.POINTER(ctypes.c_uint))[0]
            driver.memsetD32(self.ptr, value, size / 4)

    def memset32_async(self, value, size, stream):
        if size > 0:
            if isinstance(value, float):
                value = ctypes.cast((ctypes.c_float * 1)(value),
                                    ctypes.POINTER(ctypes.c_uint))[0]
            driver.memsetD32Async(self.ptr, value, size / 4)


def _malloc(size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


_alloc = _malloc


def alloc(size):
    """Calls the default allocator."""
    return _alloc(size)


def set_default_allocator(allocator=_malloc):
    global _alloc
    _alloc = allocator


class PooledMemory(Memory):

    def __init__(self, memptr, pool):
        self.ptr = memptr.mem.ptr
        self.size = memptr.mem.size
        self.pool = weakref.ref(pool)

    def __del__(self):
        if self.ptr is not None:
            self.free()

    def free(self):
        pool = self.pool()
        if pool:
            pool.free(self.ptr, self.size)
        self.ptr = None
        self.size = 0
        self.pool = None


class SingleDeviceMemoryPool(object):

    def __init__(self, allocator=_malloc):
        self._in_use = collections.defaultdict(list)
        self._free = collections.defaultdict(list)
        self._alloc = allocator

    def malloc(self, size):
        in_use = self._in_use[size]
        free = self._free[size]

        if free:
            memptr = free.pop()
        else:
            try:
                memptr = self._alloc(size)
            except runtime.CUDARuntimeError as e:
                if e.status != 2:
                    raise
                self.free_all_free()
                memptr = self._alloc(size)

        in_use.append(memptr)
        mem = PooledMemory(memptr, self)
        return MemoryPointer(mem, 0)

    def free(self, ptr, size):
        in_use = self._in_use[size]
        free = self._free[size]

        for i, memptr in enumerate(in_use):
            if memptr.mem.ptr.value == ptr.value:
                del in_use[i]
                free.append(memptr)
                break
        else:
            raise RuntimeError('Cannot free out-of-pool memory')

    def free_all_free(self):
        self._free = collections.defaultdict(list)


class MemoryPool(object):

    def __init__(self, allocator=_malloc):
        self._pools = {}
        self._alloc = allocator

    def malloc(self, size):
        dev = device.Device().id
        pool = self._pools.get(dev, None)
        if pool is None:
            pool = SingleDeviceMemoryPool(self._alloc)
            self._pools[dev] = pool
        return pool.malloc(size)

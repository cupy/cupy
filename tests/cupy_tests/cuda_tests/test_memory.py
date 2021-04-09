import ctypes
import gc
import pickle
import threading
import unittest

import fastrlock
import pytest

import cupy.cuda
from cupy.cuda import device
from cupy.cuda import memory
from cupy.cuda import stream as stream_module
from cupy import testing


class MockMemory(memory.Memory):
    cur_ptr = 1

    def __init__(self, size):
        self.ptr = MockMemory.cur_ptr
        MockMemory.cur_ptr += size
        self.size = size
        self.device_id = 0

    def __del__(self):
        self.ptr = 0
        pass


def mock_alloc(size):
    mem = MockMemory(size)
    return memory.MemoryPointer(mem, 0)


class TestUnownedMemoryClass(unittest.TestCase):

    def test_inherits_base_memory(self):
        assert issubclass(memory.UnownedMemory, memory.BaseMemory)


@testing.parameterize(*testing.product({
    'allocator': [memory._malloc, memory.malloc_managed, memory.malloc_async],
    'specify_device_id': [True, False],
}))
@testing.gpu
class TestUnownedMemory(unittest.TestCase):

    def check(self, device_id):
        if cupy.cuda.runtime.is_hip:
            if self.allocator is memory.malloc_managed:
                raise unittest.SkipTest('HIP does not support managed memory')
            if self.allocator is memory.malloc_async:
                raise unittest.SkipTest('HIP does not support async mempool')
        elif cupy.cuda.driver.get_build_version() < 11020:
            raise unittest.SkipTest('malloc_async is supported since '
                                    'CUDA 11.2')

        size = 24
        shape = (2, 3)
        dtype = cupy.float32
        with device.Device(device_id):
            src_mem_ptr = self.allocator(size)
        src_ptr = src_mem_ptr.ptr

        args = (src_ptr, size, src_mem_ptr)
        kwargs = {}
        if self.specify_device_id:
            kwargs = {'device_id': device_id}

        unowned_mem = memory.UnownedMemory(*args, **kwargs)
        assert unowned_mem.size == size
        assert unowned_mem.ptr == src_ptr
        assert unowned_mem.device_id == device_id

        arr = cupy.ndarray(shape, dtype, memory.MemoryPointer(unowned_mem, 0))

        # Delete the source object
        del src_mem_ptr

        with device.Device(device_id):
            arr[:] = 2
            assert (arr == 2).all()

    def test_device0(self):
        self.check(0)

    @testing.multi_gpu(2)
    def test_device1(self):
        self.check(1)


@testing.gpu
class TestMemoryPointer(unittest.TestCase):

    def test_int(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(1)
        assert pval == int(memptr)

    def test_add(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8)

        memptr2 = memptr + 4
        assert isinstance(memptr2, memory.MemoryPointer)
        assert pval + 4 == int(memptr2)

        memptr3 = 4 + memptr
        assert isinstance(memptr3, memory.MemoryPointer)
        assert pval + 4 == int(memptr3)

        memptr += 4
        assert isinstance(memptr, memory.MemoryPointer)
        assert pval + 4 == int(memptr)

    def test_sub(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8) + 4

        memptr2 = memptr - 4
        assert isinstance(memptr2, memory.MemoryPointer)
        assert pval == int(memptr2)

        memptr -= 4
        assert isinstance(memptr, memory.MemoryPointer)
        assert pval == int(memptr)

    def test_copy_to_and_from_host(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from(ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 4)

        b_cpu = ctypes.c_int()
        a_gpu.copy_to_host(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p), 4)
        assert b_cpu.value == a_cpu.value

    def test_copy_from_device(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from(ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 4)

        b_gpu = memory.alloc(4)
        b_gpu.copy_from(a_gpu, 4)
        b_cpu = ctypes.c_int()
        b_gpu.copy_to_host(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p), 4)
        assert b_cpu.value == a_cpu.value

    def test_copy_to_and_from_host_using_raw_ptr(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_cpu_ptr = ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p)
        a_gpu.copy_from(a_cpu_ptr.value, 4)

        b_cpu = ctypes.c_int()
        b_cpu_ptr = ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p)
        a_gpu.copy_to_host(b_cpu_ptr.value, 4)
        assert b_cpu.value == a_cpu.value

    def test_copy_from_device_using_raw_ptr(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_cpu_ptr = ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p)
        a_gpu.copy_from(a_cpu_ptr.value, 4)

        b_gpu = memory.alloc(4)
        b_gpu.copy_from(a_gpu, 4)
        b_cpu = ctypes.c_int()
        b_cpu_ptr = ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p)
        b_gpu.copy_to_host(b_cpu_ptr.value, 4)
        assert b_cpu.value == a_cpu.value

    def test_memset(self):
        a_gpu = memory.alloc(4)
        a_gpu.memset(1, 4)
        a_cpu = ctypes.c_ubyte()
        for i in range(4):
            a_gpu.copy_to_host(
                ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 1)
            assert a_cpu.value == 1
            a_gpu += 1


@testing.parameterize(*testing.product({
    'use_streams': [True, False],
}))
@testing.gpu
class TestMemoryPointerAsync(unittest.TestCase):

    def setUp(self):
        self.stream = stream_module.Stream() if self.use_streams else None

    def test_copy_to_and_from_host_async(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from_async(ctypes.cast(ctypes.byref(
            a_cpu), ctypes.c_void_p), 4, stream=self.stream)

        b_cpu = ctypes.c_int()
        a_gpu.copy_to_host_async(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p),
            4, stream=self.stream)
        if self.stream is not None:
            self.stream.synchronize()
        else:
            stream_module.get_current_stream().synchronize()
        assert b_cpu.value == a_cpu.value

    def test_copy_from_device_async(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from_async(ctypes.cast(ctypes.byref(
            a_cpu), ctypes.c_void_p), 4, stream=self.stream)

        b_gpu = memory.alloc(4)
        b_gpu.copy_from_async(a_gpu, 4, stream=self.stream)
        b_cpu = ctypes.c_int()
        b_gpu.copy_to_host_async(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p),
            4, stream=self.stream)
        if self.stream is not None:
            self.stream.synchronize()
        else:
            stream_module.get_current_stream().synchronize()
        assert b_cpu.value == a_cpu.value

    def test_copy_to_and_from_host_async_using_raw_ptr(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_cpu_ptr = ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p)
        a_gpu.copy_from_async(a_cpu_ptr.value, 4, stream=self.stream)

        b_cpu = ctypes.c_int()
        b_cpu_ptr = ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p)
        a_gpu.copy_to_host_async(b_cpu_ptr.value, 4, stream=self.stream)
        if self.stream is not None:
            self.stream.synchronize()
        else:
            stream_module.get_current_stream().synchronize()
        assert b_cpu.value == a_cpu.value

    def test_copy_from_device_async_using_raw_ptr(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_cpu_ptr = ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p)
        a_gpu.copy_from_async(a_cpu_ptr.value, 4, stream=self.stream)

        b_gpu = memory.alloc(4)
        b_gpu.copy_from_async(a_gpu, 4, stream=self.stream)
        b_cpu = ctypes.c_int()
        b_cpu_ptr = ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p)
        b_gpu.copy_to_host_async(b_cpu_ptr.value, 4, stream=self.stream)
        if self.stream is not None:
            self.stream.synchronize()
        else:
            stream_module.get_current_stream().synchronize()
        assert b_cpu.value == a_cpu.value


# -----------------------------------------------------------------------------
# Memory pool

@testing.gpu
class TestSingleDeviceMemoryPool(unittest.TestCase):

    def setUp(self):
        self.pool = memory.SingleDeviceMemoryPool(allocator=mock_alloc)
        self.unit = memory._allocation_unit_size
        self.stream = stream_module.Stream()
        self.stream_ident = self.stream.ptr

    def test_round_size(self):
        assert memory._round_size(self.unit - 1) == self.unit
        assert memory._round_size(self.unit) == self.unit
        assert memory._round_size(self.unit + 1) == self.unit * 2

    def test_bin_index_from_size(self):
        assert memory._bin_index_from_size(self.unit - 1) == 0
        assert memory._bin_index_from_size(self.unit) == 0
        assert memory._bin_index_from_size(self.unit + 1) == 1

    def test_split(self):
        mem = MockMemory(self.unit * 4)
        chunk = memory._Chunk(mem, 0, mem.size, self.stream_ident)
        tail = chunk.split(self.unit * 2)
        assert chunk.ptr() == mem.ptr
        assert chunk.offset == 0
        assert chunk.size == self.unit * 2
        assert chunk.prev is None
        assert chunk.next.ptr() == tail.ptr()
        assert chunk.stream_ident == self.stream_ident
        assert tail.ptr() == mem.ptr + self.unit * 2
        assert tail.offset == self.unit * 2
        assert tail.size == self.unit * 2
        assert tail.prev.ptr() == chunk.ptr()
        assert tail.next is None
        assert tail.stream_ident == self.stream_ident

        tail_of_head = chunk.split(self.unit)
        assert chunk.ptr() == mem.ptr
        assert chunk.offset == 0
        assert chunk.size == self.unit
        assert chunk.prev is None
        assert chunk.next.ptr() == tail_of_head.ptr()
        assert chunk.stream_ident == self.stream_ident
        assert tail_of_head.ptr() == mem.ptr + self.unit
        assert tail_of_head.offset == self.unit
        assert tail_of_head.size == self.unit
        assert tail_of_head.prev.ptr() == chunk.ptr()
        assert tail_of_head.next.ptr() == tail.ptr()
        assert tail_of_head.stream_ident == self.stream_ident

        tail_of_tail = tail.split(self.unit)
        assert tail.ptr() == chunk.ptr() + self.unit * 2
        assert tail.offset == self.unit * 2
        assert tail.size == self.unit
        assert tail.prev.ptr() == tail_of_head.ptr()
        assert tail.next.ptr() == tail_of_tail.ptr()
        assert tail.stream_ident == self.stream_ident
        assert tail_of_tail.ptr() == mem.ptr + self.unit * 3
        assert tail_of_tail.offset == self.unit * 3
        assert tail_of_tail.size == self.unit
        assert tail_of_tail.prev.ptr() == tail.ptr()
        assert tail_of_tail.next is None
        assert tail_of_tail.stream_ident == self.stream_ident

    def test_merge(self):
        mem = MockMemory(self.unit * 4)
        chunk = memory._Chunk(mem, 0, mem.size, self.stream_ident)
        chunk_ptr = chunk.ptr()
        chunk_offset = chunk.offset
        chunk_size = chunk.size

        tail = chunk.split(self.unit * 2)
        head = chunk
        head_ptr = head.ptr()
        head_offset = head.offset
        head_size = head.size
        tail_ptr = tail.ptr()
        tail_offset = tail.offset
        tail_size = tail.size

        tail_of_head = head.split(self.unit)
        tail_of_tail = tail.split(self.unit)

        head.merge(tail_of_head)
        assert head.ptr() == head_ptr
        assert head.offset == head_offset
        assert head.size == head_size
        assert head.prev is None
        assert head.next.ptr() == tail_ptr
        assert head.stream_ident == self.stream_ident

        tail.merge(tail_of_tail)
        assert tail.ptr() == tail_ptr
        assert tail.offset == tail_offset
        assert tail.size == tail_size
        assert tail.prev.ptr() == head_ptr
        assert tail.next is None
        assert tail.stream_ident == self.stream_ident

        head.merge(tail)
        assert head.ptr() == chunk_ptr
        assert head.offset == chunk_offset
        assert head.size == chunk_size
        assert head.prev is None
        assert head.next is None
        assert head.stream_ident == self.stream_ident

    def test_alloc(self):
        p1 = self.pool.malloc(self.unit * 4)
        p2 = self.pool.malloc(self.unit * 4)
        p3 = self.pool.malloc(self.unit * 8)
        assert p1.ptr != p2.ptr
        assert p1.ptr != p3.ptr
        assert p2.ptr != p3.ptr

    def test_alloc_split(self):
        p = self.pool.malloc(self.unit * 4)
        ptr = p.ptr
        del p
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        assert ptr == head.ptr
        assert ptr + self.unit * 2 == tail.ptr

    def test_alloc_limit(self):
        self.pool.set_limit(size=(self.unit * 6))

        p1 = self.pool.malloc(self.unit * 5)
        p2 = self.pool.malloc(self.unit * 1)
        with self.assertRaises(memory.OutOfMemoryError):
            self.pool.malloc(self.unit)

        self.pool.set_limit(size=(self.unit * 7))
        p3 = self.pool.malloc(self.unit)
        del p1, p2, p3

    def test_free(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(self.unit * 4)
        assert ptr1 == p2.ptr

    def test_free_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 4)
        assert ptr1 != p2.ptr

    def test_free_merge(self):
        p = self.pool.malloc(self.unit * 4)
        ptr = p.ptr
        del p

        # merge head into tail
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        assert ptr == head.ptr
        del tail
        del head
        p = self.pool.malloc(self.unit * 4)
        assert ptr == p.ptr
        del p

        # merge tail into head
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        assert ptr == head.ptr
        del head
        del tail
        p = self.pool.malloc(self.unit * 4)
        assert ptr == p.ptr
        del p

    def test_free_different_size(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(self.unit * 8)
        assert ptr1 != p2.ptr

    def test_free_all_blocks(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        self.pool.free_all_blocks()
        p2 = self.pool.malloc(self.unit * 4)
        assert ptr1 != p2.ptr
        del p2

    def test_free_all_blocks_split(self):
        # do not free splitted blocks
        p = self.pool.malloc(self.unit * 4)
        del p
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        tailptr = tail.ptr
        del tail
        self.pool.free_all_blocks()
        p = self.pool.malloc(self.unit * 2)
        assert tailptr == p.ptr
        del head

    def test_free_all_blocks_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 4)
            ptr2 = p2.ptr
            del p2
        self.pool.free_all_blocks(stream=stream_module.Stream.null)
        p3 = self.pool.malloc(self.unit * 4)
        assert ptr1 != p3.ptr
        assert ptr2 != p3.ptr
        with self.stream:
            p4 = self.pool.malloc(self.unit * 4)
            assert ptr1 != p4.ptr
            assert ptr2 == p4.ptr

    def test_free_all_blocks_all_streams(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 4)
            ptr2 = p2.ptr
            del p2
        self.pool.free_all_blocks()
        p3 = self.pool.malloc(self.unit * 4)
        assert ptr1 != p3.ptr
        assert ptr2 != p3.ptr
        with self.stream:
            p4 = self.pool.malloc(self.unit * 4)
            assert ptr1 != p4.ptr
            assert ptr2 != p4.ptr

    def test_free_all_free(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        with testing.assert_warns(DeprecationWarning):
            self.pool.free_all_free()
        p2 = self.pool.malloc(self.unit * 4)
        assert ptr1 != p2.ptr

    def test_used_bytes(self):
        p1 = self.pool.malloc(self.unit * 2)
        assert self.unit * 2 == self.pool.used_bytes()
        p2 = self.pool.malloc(self.unit * 4)
        assert self.unit * 6 == self.pool.used_bytes()
        del p2
        assert self.unit * 2 == self.pool.used_bytes()
        del p1
        assert self.unit * 0 == self.pool.used_bytes()
        p3 = self.pool.malloc(self.unit * 1)
        assert self.unit * 1 == self.pool.used_bytes()
        del p3

    def test_used_bytes_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 2)
        assert self.unit * 2 == self.pool.used_bytes()
        del p2

    def test_free_bytes(self):
        p1 = self.pool.malloc(self.unit * 2)
        assert self.unit * 0 == self.pool.free_bytes()
        p2 = self.pool.malloc(self.unit * 4)
        assert self.unit * 0 == self.pool.free_bytes()
        del p2
        assert self.unit * 4 == self.pool.free_bytes()
        del p1
        assert self.unit * 6 == self.pool.free_bytes()
        p3 = self.pool.malloc(self.unit * 1)
        assert self.unit * 5 == self.pool.free_bytes()
        del p3

    def test_free_bytes_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 2)
        assert self.unit * 4 == self.pool.free_bytes()
        del p2

    def test_total_bytes(self):
        p1 = self.pool.malloc(self.unit * 2)
        assert self.unit * 2 == self.pool.total_bytes()
        p2 = self.pool.malloc(self.unit * 4)
        assert self.unit * 6 == self.pool.total_bytes()
        del p1
        assert self.unit * 6 == self.pool.total_bytes()
        del p2
        assert self.unit * 6 == self.pool.total_bytes()
        p3 = self.pool.malloc(self.unit * 1)
        assert self.unit * 6 == self.pool.total_bytes()

        assert (self.pool.used_bytes() + self.pool.free_bytes()
                == self.pool.total_bytes())

        del p3

        self.pool.free_all_blocks()
        assert 0 == self.pool.total_bytes()

    def test_total_bytes_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 2)
        assert self.unit * 6 == self.pool.total_bytes()
        del p2

    def test_get_limit(self):
        # limit is disabled by default
        assert 0 == self.pool.get_limit()

    def test_set_limit_size(self):
        self.pool.set_limit(size=1024)
        assert 1024 == self.pool.get_limit()

        self.pool.set_limit(size=2**33)
        assert 2**33 == self.pool.get_limit()

        self.pool.set_limit(size=0)
        assert 0 == self.pool.get_limit()

        with self.assertRaises(ValueError):
            self.pool.set_limit(size=-1)

    def test_set_limit_fraction(self):
        _, total = cupy.cuda.runtime.memGetInfo()

        self.pool.set_limit(fraction=0)
        assert 0 == self.pool.get_limit()

        self.pool.set_limit(fraction=0.5)
        assert total * 0.5 == self.pool.get_limit()

        self.pool.set_limit(fraction=1.0)
        assert total == self.pool.get_limit()

        with self.assertRaises(ValueError):
            self.pool.set_limit(fraction=-1)

        with self.assertRaises(ValueError):
            self.pool.set_limit(fraction=1.1)

    def test_parse_limit_string(self):
        parse_limit_string = memory._parse_limit_string

        # size
        param = parse_limit_string('0')
        assert 0 == param['size']
        assert None is param['fraction']

        param = parse_limit_string('1073741824')
        assert 1073741824 == param['size']
        assert None is param['fraction']

        # fraction
        param = parse_limit_string('0%')
        assert None is param['size']
        assert 0.0 == param['fraction']

        param = parse_limit_string('40%')
        assert None is param['size']
        assert 0.4 == param['fraction']

        param = parse_limit_string('70.5%')
        assert None is param['size']
        assert 0.705 == param['fraction']

        param = parse_limit_string('100%')
        assert None is param['size']
        assert 1.0 == param['fraction']


@testing.parameterize(*testing.product({
    'allocator': [memory._malloc, memory.malloc_managed],
}))
@testing.gpu
class TestMemoryPool(unittest.TestCase):

    def setUp(self):
        if (cupy.cuda.runtime.is_hip
                and self.allocator is memory.malloc_managed):
            raise unittest.SkipTest('HIP does not support managed memory')
        self.pool = memory.MemoryPool(self.allocator)

    def tearDown(self):
        self.pool.free_all_blocks()

    def test_zero_size_alloc(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(0).mem
            assert isinstance(mem, memory.Memory)
            assert not isinstance(mem, memory.PooledMemory)

    def test_double_free(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(1).mem
            mem.free()
            mem.free()

    def test_free_all_blocks(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(1).mem
            assert isinstance(mem, memory.BaseMemory)
            assert isinstance(mem, memory.PooledMemory)
            assert self.pool.n_free_blocks() == 0
            mem.free()
            assert self.pool.n_free_blocks() == 1
            self.pool.free_all_blocks()
            assert self.pool.n_free_blocks() == 0

    def test_free_all_blocks_without_malloc(self):
        with cupy.cuda.Device():
            # call directly without malloc.
            self.pool.free_all_blocks()
            assert self.pool.n_free_blocks() == 0

    def test_free_all_free(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(1).mem
            assert isinstance(mem, memory.BaseMemory)
            assert isinstance(mem, memory.PooledMemory)
            assert self.pool.n_free_blocks() == 0
            mem.free()
            assert self.pool.n_free_blocks() == 1
            with testing.assert_warns(DeprecationWarning):
                self.pool.free_all_free()
            assert self.pool.n_free_blocks() == 0

    def test_free_all_free_without_malloc(self):
        with cupy.cuda.Device():
            # call directly without malloc.
            with testing.assert_warns(DeprecationWarning):
                self.pool.free_all_free()
            assert self.pool.n_free_blocks() == 0

    def test_n_free_blocks_without_malloc(self):
        with cupy.cuda.Device():
            # call directly without malloc/free_all_free.
            assert self.pool.n_free_blocks() == 0

    def test_used_bytes(self):
        with cupy.cuda.Device():
            assert 0 == self.pool.used_bytes()

    def test_free_bytes(self):
        with cupy.cuda.Device():
            assert 0 == self.pool.free_bytes()

    def test_total_bytes(self):
        with cupy.cuda.Device():
            assert 0 == self.pool.total_bytes()


@testing.gpu
class TestAllocator(unittest.TestCase):

    def setUp(self):
        self.old_pool = cupy.get_default_memory_pool()
        self.pool = memory.MemoryPool()
        memory.set_allocator(self.pool.malloc)

    def tearDown(self):
        self.pool.free_all_blocks()
        memory.set_allocator(self.old_pool.malloc)

    def test_set_allocator(self):
        with cupy.cuda.Device():
            assert 0 == self.pool.used_bytes()
            arr = cupy.arange(128, dtype=cupy.int64)
            assert 1024 == arr.data.mem.size
            assert 1024 == self.pool.used_bytes()

    def test_get_allocator(self):
        assert memory.get_allocator() == self.pool.malloc

    def test_allocator_context_manager(self):
        new_pool = memory.MemoryPool()
        with cupy.cuda.using_allocator(new_pool.malloc):
            assert memory.get_allocator() == new_pool.malloc
        assert memory.get_allocator() == self.pool.malloc

    def test_set_allocator_cm(self):
        new_pool = memory.MemoryPool()
        new_pool2 = memory.MemoryPool()
        with cupy.cuda.using_allocator(new_pool.malloc):
            with self.assertRaises(ValueError):
                memory.set_allocator(new_pool2.malloc)

    def test_allocator_nested_context_manager(self):
        new_pool = memory.MemoryPool()
        with cupy.cuda.using_allocator(new_pool.malloc):
            new_pool2 = memory.MemoryPool()
            assert memory.get_allocator() == new_pool.malloc
            with cupy.cuda.using_allocator(new_pool2.malloc):
                assert memory.get_allocator() == new_pool2.malloc
            assert memory.get_allocator() == new_pool.malloc
        assert memory.get_allocator() == self.pool.malloc

    def test_allocator_thread_local(self):
        def thread_body(self):
            new_pool = memory.MemoryPool()
            with cupy.cuda.using_allocator(new_pool.malloc):
                assert memory.get_allocator() == new_pool.malloc
                threading.Barrier(2)
                arr = cupy.zeros(128, dtype=cupy.int64)
                threading.Barrier(2)
                assert arr.data.mem.size == new_pool.used_bytes()
                threading.Barrier(2)
            assert memory.get_allocator() == self.pool.malloc

        with cupy.cuda.Device():
            t = threading.Thread(target=thread_body, args=(self,))
            t.daemon = True
            t.start()
            threading.Barrier(2)
            assert memory.get_allocator() == self.pool.malloc
            arr = cupy.ones(256, dtype=cupy.int64)
            threading.Barrier(2)
            assert arr.data.mem.size == self.pool.used_bytes()
            threading.Barrier(2)
            t.join()

    def test_thread_local_valid(self):
        new_pool = memory.MemoryPool()
        arr = None
        with cupy.cuda.using_allocator(new_pool.malloc):
            arr = cupy.zeros(128, dtype=cupy.int64)
            arr += 1
        # Check that arr and the pool have not ben released
        assert arr.data.mem.size == new_pool.used_bytes()
        assert arr.sum() == 128

    def _reuse_between_thread(self, stream_main, stream_sub):
        new_pool = memory.MemoryPool()

        def job(stream):
            with cupy.cuda.using_allocator(new_pool.malloc):
                with stream:
                    arr = cupy.arange(16)
            self._ptr = arr.data.ptr
            del arr
            self._error = False

        # Run in main thread.
        self._ptr = -1
        self._error = True
        job(stream_main)
        assert not self._error
        main_ptr = self._ptr

        # Run in sub thread.
        self._ptr = -1
        self._error = True
        with cupy.cuda.Device():
            t = threading.Thread(target=job, args=(stream_sub,))
            t.daemon = True
            t.start()
            t.join()
        assert not self._error
        return main_ptr, self._ptr

    def test_reuse_between_thread(self):
        stream = cupy.cuda.Stream.null
        main_ptr, sub_ptr = self._reuse_between_thread(stream, stream)
        assert main_ptr == sub_ptr

    def test_reuse_between_thread_same_stream(self):
        stream = cupy.cuda.Stream()
        main_ptr, sub_ptr = self._reuse_between_thread(stream, stream)
        assert main_ptr == sub_ptr

    def test_reuse_between_thread_different_stream(self):
        stream1 = cupy.cuda.Stream()
        stream2 = cupy.cuda.Stream()
        main_ptr, sub_ptr = self._reuse_between_thread(stream1, stream2)
        assert main_ptr != sub_ptr

    @pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason='No PTDS on HIP')
    def test_reuse_between_thread_ptds(self):
        stream = cupy.cuda.Stream.ptds
        main_ptr, sub_ptr = self._reuse_between_thread(stream, stream)
        assert main_ptr != sub_ptr


@testing.gpu
class TestAllocatorDisabled(unittest.TestCase):

    def setUp(self):
        self.pool = cupy.get_default_memory_pool()

    def tearDown(self):
        memory.set_allocator(self.pool.malloc)

    def _check_pool_not_used(self):
        used_bytes = self.pool.used_bytes()
        with cupy.cuda.Device():
            arr = cupy.arange(128, dtype=cupy.int64)
            assert 0 == self.pool.used_bytes() - used_bytes
            del arr

    def test(self):
        memory.set_allocator()
        self._check_pool_not_used()

    def test_none(self):
        memory.set_allocator(None)
        self._check_pool_not_used()


class PythonAllocator(object):
    def __init__(self):
        self.malloc_called = False
        self.free_called = False

    def malloc(self, size, device_id):
        self.malloc_called = True
        return cupy.cuda.runtime.malloc(size)

    def free(self, size, device_id):
        self.free_called = True
        cupy.cuda.runtime.free(size)


@testing.gpu
class TestPythonFunctionAllocator(unittest.TestCase):
    def setUp(self):
        self.old_pool = cupy.get_default_memory_pool()
        self.alloc = PythonAllocator()
        python_alloc = memory.PythonFunctionAllocator(
            self.alloc.malloc, self.alloc.free)
        memory.set_allocator(python_alloc.malloc)

    def tearDown(self):
        memory.set_allocator(self.old_pool.malloc)

    def test_allocator(self):
        assert not self.alloc.malloc_called and not self.alloc.free_called
        cupy.zeros(10)
        assert self.alloc.malloc_called and self.alloc.free_called


@testing.gpu
class TestMemInfo(unittest.TestCase):

    def test_mem_info(self):
        d = cupy.cuda.Device()
        mem_info = d.mem_info
        assert isinstance(mem_info, tuple)
        assert len(mem_info) == 2
        assert all(isinstance(m, int) for m in mem_info)
        assert all(m > 0 for m in mem_info)


@testing.gpu
class TestLockAndNoGc(unittest.TestCase):

    def test(self):
        lock = fastrlock.rlock.FastRLock()
        ctx = memory.LockAndNoGc(lock)

        assert gc.isenabled()
        self.assertRaises(Exception, lock.release)
        with ctx:
            assert not gc.isenabled()
            lock.release()
            lock.acquire()
        assert gc.isenabled()
        self.assertRaises(Exception, lock.release)


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = memory.OutOfMemoryError(124, 1024, 1024)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support async allocator')
@pytest.mark.skipif(cupy.cuda.driver.get_build_version() < 11020,
                    reason='malloc_async is supported since CUDA 11.2')
class TestMallocAsync(unittest.TestCase):

    def setUp(self):
        self.old_pool = cupy.get_default_memory_pool()
        memory.set_allocator(memory.malloc_async)

    def tearDown(self):
        memory.set_allocator(self.old_pool.malloc)

    def _check_pool_not_used(self):
        used_bytes = self.old_pool.used_bytes()
        with cupy.cuda.Device():
            arr = cupy.arange(128, dtype=cupy.int64)
            assert 0 == self.old_pool.used_bytes() - used_bytes
            del arr

    def test(self):
        self._check_pool_not_used()

    def test_stream1(self):
        # Check: pool is not used when on a stream
        s = cupy.cuda.Stream()
        with s:
            self._check_pool_not_used()

    def test_stream2(self):
        # Check: the memory was allocated on the right stream
        s = cupy.cuda.Stream()
        with s:
            memptr = memory.alloc(100)
            assert memptr.mem.stream == s.ptr

    def test_stream3(self):
        # Check: destory stream does not affect memory deallocation
        s = cupy.cuda.Stream()
        with s:
            memptr = memory.alloc(100)

        del s
        gc.collect()
        del memptr

    def test_stream4(self):
        # Check: free on the same stream
        s = cupy.cuda.Stream()
        with s:
            memptr = memory.alloc(100)
            del memptr

    def test_stream5(self):
        # Check: free on another stream
        s1 = cupy.cuda.Stream()
        with s1:
            memptr = memory.alloc(100)
        del s1

        s2 = cupy.cuda.Stream()
        with s2:
            del memptr


@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support async allocator')
@pytest.mark.skipif(cupy.cuda.driver.get_build_version() < 11020,
                    reason='malloc_async is supported since CUDA 11.2')
class TestMemoryAsyncPool(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryAsyncPool()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.Device().synchronize()

    def test_zero_size_alloc(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(0).mem
            assert isinstance(mem, memory.MemoryAsync)
            assert not isinstance(mem, memory.PooledMemory)

    def test_alloc(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(100).mem
            assert isinstance(mem, memory.MemoryAsync)
            assert not isinstance(mem, memory.PooledMemory)

    @testing.slow
    def test_alloc_large_chunk(self):
        self.pool.free_all_blocks()
        with cupy.cuda.Device() as d:
            _, mem_total = d.mem_info
            mem = self.pool.malloc(int(0.7 * mem_total)).mem  # 70% memory
            del mem
            mem = self.pool.malloc(int(0.3 * mem_total)).mem  # 30% memory # noqa

    def test_free_all_blocks(self):
        with cupy.cuda.Device():
            mem = self.pool.malloc(1).mem
            del mem
            self.pool.free_all_blocks()

    @testing.slow
    def test_free_all_blocks_large_chunk(self):
        # When memory is returned to the async mempool, it is not immediately
        # visible to normal malloc routines until after a sync happens.
        default_pool = cupy.get_default_memory_pool()
        with cupy.cuda.Device() as d:
            _, mem_total = d.mem_info
            mem = self.pool.malloc(int(0.7 * mem_total)).mem  # 70% memory
            del mem
            with pytest.raises(memory.OutOfMemoryError):
                default_pool.malloc(int(0.3 * mem_total))  # 30% memory
            self.pool.free_all_blocks()  # synchronize
            default_pool.malloc(int(0.3 * mem_total))  # this time it'd work

    @testing.slow
    def test_interaction_with_CuPy_default_pool(self):
        # Test saneness of cudaMallocAsync
        default_pool = cupy.get_default_memory_pool()
        with cupy.cuda.Device() as d:
            _, mem_total = d.mem_info
            mem = default_pool.malloc(int(0.7 * mem_total)).mem  # 70% memory
            del mem
            with pytest.raises(memory.OutOfMemoryError):
                self.pool.malloc(int(0.3 * mem_total))  # 30% memory
            default_pool.free_all_blocks()
            self.pool.malloc(int(0.3 * mem_total))  # this time it'd work

import ctypes
import gc
import pickle
import threading
import unittest

import fastrlock

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
    'allocator': [memory._malloc, memory.malloc_managed],
    'specify_device_id': [True, False],
}))
@testing.gpu
class TestUnownedMemory(unittest.TestCase):

    def check(self, device_id):
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
        self.assertEqual(pval, int(memptr))

    def test_add(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8)

        memptr2 = memptr + 4
        self.assertIsInstance(memptr2, memory.MemoryPointer)
        self.assertEqual(pval + 4, int(memptr2))

        memptr3 = 4 + memptr
        self.assertIsInstance(memptr3, memory.MemoryPointer)
        self.assertEqual(pval + 4, int(memptr3))

        memptr += 4
        self.assertIsInstance(memptr, memory.MemoryPointer)
        self.assertEqual(pval + 4, int(memptr))

    def test_sub(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8) + 4

        memptr2 = memptr - 4
        self.assertIsInstance(memptr2, memory.MemoryPointer)
        self.assertEqual(pval, int(memptr2))

        memptr -= 4
        self.assertIsInstance(memptr, memory.MemoryPointer)
        self.assertEqual(pval, int(memptr))

    def test_copy_to_and_from_host(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from(ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 4)
        b_cpu = ctypes.c_int()
        a_gpu.copy_to_host(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p), 4)
        self.assertEqual(b_cpu.value, a_cpu.value)

    def test_copy_from_device(self):
        a_gpu = memory.alloc(4)
        a_cpu = ctypes.c_int(100)
        a_gpu.copy_from(ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 4)

        b_gpu = memory.alloc(4)
        b_gpu.copy_from(a_gpu, 4)
        b_cpu = ctypes.c_int()
        b_gpu.copy_to_host(
            ctypes.cast(ctypes.byref(b_cpu), ctypes.c_void_p), 4)
        self.assertEqual(b_cpu.value, a_cpu.value)

    def test_memset(self):
        a_gpu = memory.alloc(4)
        a_gpu.memset(1, 4)
        a_cpu = ctypes.c_ubyte()
        for i in range(4):
            a_gpu.copy_to_host(
                ctypes.cast(ctypes.byref(a_cpu), ctypes.c_void_p), 1)
            self.assertEqual(a_cpu.value, 1)
            a_gpu += 1


# -----------------------------------------------------------------------------
# Memory pool

@testing.gpu
class TestSingleDeviceMemoryPool(unittest.TestCase):

    def setUp(self):
        self.pool = memory.SingleDeviceMemoryPool(allocator=mock_alloc)
        self.unit = memory._allocation_unit_size
        self.stream = stream_module.Stream()
        self.stream_ptr = self.stream.ptr

    def test_round_size(self):
        self.assertEqual(memory._round_size(self.unit - 1), self.unit)
        self.assertEqual(memory._round_size(self.unit), self.unit)
        self.assertEqual(memory._round_size(self.unit + 1), self.unit * 2)

    def test_bin_index_from_size(self):
        self.assertEqual(memory._bin_index_from_size(self.unit - 1), 0)
        self.assertEqual(memory._bin_index_from_size(self.unit), 0)
        self.assertEqual(memory._bin_index_from_size(self.unit + 1), 1)

    def test_split(self):
        mem = MockMemory(self.unit * 4)
        chunk = memory._Chunk(mem, 0, mem.size, self.stream_ptr)
        tail = chunk.split(self.unit * 2)
        self.assertEqual(chunk.ptr(), mem.ptr)
        self.assertEqual(chunk.offset, 0)
        self.assertEqual(chunk.size, self.unit * 2)
        self.assertEqual(chunk.prev, None)
        self.assertEqual(chunk.next.ptr(), tail.ptr())
        self.assertEqual(chunk.stream_ptr, self.stream_ptr)
        self.assertEqual(tail.ptr(), mem.ptr + self.unit * 2)
        self.assertEqual(tail.offset, self.unit * 2)
        self.assertEqual(tail.size, self.unit * 2)
        self.assertEqual(tail.prev.ptr(), chunk.ptr())
        self.assertEqual(tail.next, None)
        self.assertEqual(tail.stream_ptr, self.stream_ptr)

        tail_of_head = chunk.split(self.unit)
        self.assertEqual(chunk.ptr(), mem.ptr)
        self.assertEqual(chunk.offset, 0)
        self.assertEqual(chunk.size, self.unit)
        self.assertEqual(chunk.prev, None)
        self.assertEqual(chunk.next.ptr(), tail_of_head.ptr())
        self.assertEqual(chunk.stream_ptr, self.stream_ptr)
        self.assertEqual(tail_of_head.ptr(), mem.ptr + self.unit)
        self.assertEqual(tail_of_head.offset, self.unit)
        self.assertEqual(tail_of_head.size, self.unit)
        self.assertEqual(tail_of_head.prev.ptr(), chunk.ptr())
        self.assertEqual(tail_of_head.next.ptr(), tail.ptr())
        self.assertEqual(tail_of_head.stream_ptr, self.stream_ptr)

        tail_of_tail = tail.split(self.unit)
        self.assertEqual(tail.ptr(), chunk.ptr() + self.unit * 2)
        self.assertEqual(tail.offset, self.unit * 2)
        self.assertEqual(tail.size, self.unit)
        self.assertEqual(tail.prev.ptr(), tail_of_head.ptr())
        self.assertEqual(tail.next.ptr(), tail_of_tail.ptr())
        self.assertEqual(tail.stream_ptr, self.stream_ptr)
        self.assertEqual(tail_of_tail.ptr(), mem.ptr + self.unit * 3)
        self.assertEqual(tail_of_tail.offset, self.unit * 3)
        self.assertEqual(tail_of_tail.size, self.unit)
        self.assertEqual(tail_of_tail.prev.ptr(), tail.ptr())
        self.assertEqual(tail_of_tail.next, None)
        self.assertEqual(tail_of_tail.stream_ptr, self.stream_ptr)

    def test_merge(self):
        mem = MockMemory(self.unit * 4)
        chunk = memory._Chunk(mem, 0, mem.size, self.stream_ptr)
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
        self.assertEqual(head.ptr(), head_ptr)
        self.assertEqual(head.offset, head_offset)
        self.assertEqual(head.size, head_size)
        self.assertEqual(head.prev, None)
        self.assertEqual(head.next.ptr(), tail_ptr)
        self.assertEqual(head.stream_ptr, self.stream_ptr)

        tail.merge(tail_of_tail)
        self.assertEqual(tail.ptr(), tail_ptr)
        self.assertEqual(tail.offset, tail_offset)
        self.assertEqual(tail.size, tail_size)
        self.assertEqual(tail.prev.ptr(), head_ptr)
        self.assertEqual(tail.next, None)
        self.assertEqual(tail.stream_ptr, self.stream_ptr)

        head.merge(tail)
        self.assertEqual(head.ptr(), chunk_ptr)
        self.assertEqual(head.offset, chunk_offset)
        self.assertEqual(head.size, chunk_size)
        self.assertEqual(head.prev, None)
        self.assertEqual(head.next, None)
        self.assertEqual(head.stream_ptr, self.stream_ptr)

    def test_alloc(self):
        p1 = self.pool.malloc(self.unit * 4)
        p2 = self.pool.malloc(self.unit * 4)
        p3 = self.pool.malloc(self.unit * 8)
        self.assertNotEqual(p1.ptr, p2.ptr)
        self.assertNotEqual(p1.ptr, p3.ptr)
        self.assertNotEqual(p2.ptr, p3.ptr)

    def test_alloc_split(self):
        p = self.pool.malloc(self.unit * 4)
        ptr = p.ptr
        del p
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        self.assertEqual(ptr, head.ptr)
        self.assertEqual(ptr + self.unit * 2, tail.ptr)

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
        self.assertEqual(ptr1, p2.ptr)

    def test_free_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 4)
        self.assertNotEqual(ptr1, p2.ptr)

    def test_free_merge(self):
        p = self.pool.malloc(self.unit * 4)
        ptr = p.ptr
        del p

        # merge head into tail
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        self.assertEqual(ptr, head.ptr)
        del tail
        del head
        p = self.pool.malloc(self.unit * 4)
        self.assertEqual(ptr, p.ptr)
        del p

        # merge tail into head
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        self.assertEqual(ptr, head.ptr)
        del head
        del tail
        p = self.pool.malloc(self.unit * 4)
        self.assertEqual(ptr, p.ptr)
        del p

    def test_free_different_size(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(self.unit * 8)
        self.assertNotEqual(ptr1, p2.ptr)

    def test_free_all_blocks(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        self.pool.free_all_blocks()
        p2 = self.pool.malloc(self.unit * 4)
        self.assertNotEqual(ptr1, p2.ptr)
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
        self.assertEqual(tailptr, p.ptr)
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
        self.assertNotEqual(ptr1, p3.ptr)
        self.assertNotEqual(ptr2, p3.ptr)
        with self.stream:
            p4 = self.pool.malloc(self.unit * 4)
            self.assertNotEqual(ptr1, p4.ptr)
            self.assertEqual(ptr2, p4.ptr)

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
        self.assertNotEqual(ptr1, p3.ptr)
        self.assertNotEqual(ptr2, p3.ptr)
        with self.stream:
            p4 = self.pool.malloc(self.unit * 4)
            self.assertNotEqual(ptr1, p4.ptr)
            self.assertNotEqual(ptr2, p4.ptr)

    def test_free_all_free(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        with testing.assert_warns(DeprecationWarning):
            self.pool.free_all_free()
        p2 = self.pool.malloc(self.unit * 4)
        self.assertNotEqual(ptr1, p2.ptr)

    def test_used_bytes(self):
        p1 = self.pool.malloc(self.unit * 2)
        self.assertEqual(self.unit * 2, self.pool.used_bytes())
        p2 = self.pool.malloc(self.unit * 4)
        self.assertEqual(self.unit * 6, self.pool.used_bytes())
        del p2
        self.assertEqual(self.unit * 2, self.pool.used_bytes())
        del p1
        self.assertEqual(self.unit * 0, self.pool.used_bytes())
        p3 = self.pool.malloc(self.unit * 1)
        self.assertEqual(self.unit * 1, self.pool.used_bytes())
        del p3

    def test_used_bytes_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 2)
        self.assertEqual(self.unit * 2, self.pool.used_bytes())
        del p2

    def test_free_bytes(self):
        p1 = self.pool.malloc(self.unit * 2)
        self.assertEqual(self.unit * 0, self.pool.free_bytes())
        p2 = self.pool.malloc(self.unit * 4)
        self.assertEqual(self.unit * 0, self.pool.free_bytes())
        del p2
        self.assertEqual(self.unit * 4, self.pool.free_bytes())
        del p1
        self.assertEqual(self.unit * 6, self.pool.free_bytes())
        p3 = self.pool.malloc(self.unit * 1)
        self.assertEqual(self.unit * 5, self.pool.free_bytes())
        del p3

    def test_free_bytes_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 2)
        self.assertEqual(self.unit * 4, self.pool.free_bytes())
        del p2

    def test_total_bytes(self):
        p1 = self.pool.malloc(self.unit * 2)
        self.assertEqual(self.unit * 2, self.pool.total_bytes())
        p2 = self.pool.malloc(self.unit * 4)
        self.assertEqual(self.unit * 6, self.pool.total_bytes())
        del p1
        self.assertEqual(self.unit * 6, self.pool.total_bytes())
        del p2
        self.assertEqual(self.unit * 6, self.pool.total_bytes())
        p3 = self.pool.malloc(self.unit * 1)
        self.assertEqual(self.unit * 6, self.pool.total_bytes())

        self.assertEqual(
            self.pool.used_bytes() + self.pool.free_bytes(),
            self.pool.total_bytes())

        del p3

        self.pool.free_all_blocks()
        self.assertEqual(0, self.pool.total_bytes())

    def test_total_bytes_stream(self):
        p1 = self.pool.malloc(self.unit * 4)
        del p1
        with self.stream:
            p2 = self.pool.malloc(self.unit * 2)
        self.assertEqual(self.unit * 6, self.pool.total_bytes())
        del p2

    def test_get_limit(self):
        # limit is disabled by default
        self.assertEqual(0, self.pool.get_limit())

    def test_set_limit_size(self):
        self.pool.set_limit(size=1024)
        self.assertEqual(1024, self.pool.get_limit())

        self.pool.set_limit(size=2**33)
        self.assertEqual(2**33, self.pool.get_limit())

        self.pool.set_limit(size=0)
        self.assertEqual(0, self.pool.get_limit())

        with self.assertRaises(ValueError):
            self.pool.set_limit(size=-1)

    def test_set_limit_fraction(self):
        _, total = cupy.cuda.runtime.memGetInfo()

        self.pool.set_limit(fraction=0)
        self.assertEqual(0, self.pool.get_limit())

        self.pool.set_limit(fraction=0.5)
        self.assertEqual(total * 0.5, self.pool.get_limit())

        self.pool.set_limit(fraction=1.0)
        self.assertEqual(total, self.pool.get_limit())

        with self.assertRaises(ValueError):
            self.pool.set_limit(fraction=-1)

        with self.assertRaises(ValueError):
            self.pool.set_limit(fraction=1.1)

    def test_parse_limit_string(self):
        parse_limit_string = self.pool._parse_limit_string

        # size
        param = parse_limit_string('0')
        self.assertEqual(0, param['size'])
        self.assertEqual(None, param['fraction'])

        param = parse_limit_string('1073741824')
        self.assertEqual(1073741824, param['size'])
        self.assertEqual(None, param['fraction'])

        # fraction
        param = parse_limit_string('0%')
        self.assertEqual(None, param['size'])
        self.assertEqual(0.0, param['fraction'])

        param = parse_limit_string('40%')
        self.assertEqual(None, param['size'])
        self.assertEqual(0.4, param['fraction'])

        param = parse_limit_string('70.5%')
        self.assertEqual(None, param['size'])
        self.assertEqual(0.705, param['fraction'])

        param = parse_limit_string('100%')
        self.assertEqual(None, param['size'])
        self.assertEqual(1.0, param['fraction'])


@testing.parameterize(*testing.product({
    'allocator': [memory._malloc, memory.malloc_managed],
}))
@testing.gpu
class TestMemoryPool(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryPool(self.allocator)

    def test_zero_size_alloc(self):
        with cupy.cuda.Device(0):
            mem = self.pool.malloc(0).mem
            self.assertIsInstance(mem, memory.Memory)
            self.assertNotIsInstance(mem, memory.PooledMemory)

    def test_double_free(self):
        with cupy.cuda.Device(0):
            mem = self.pool.malloc(1).mem
            mem.free()
            mem.free()

    def test_free_all_blocks(self):
        with cupy.cuda.Device(0):
            mem = self.pool.malloc(1).mem
            self.assertIsInstance(mem, memory.BaseMemory)
            self.assertIsInstance(mem, memory.PooledMemory)
            self.assertEqual(self.pool.n_free_blocks(), 0)
            mem.free()
            self.assertEqual(self.pool.n_free_blocks(), 1)
            self.pool.free_all_blocks()
            self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_free_all_blocks_without_malloc(self):
        with cupy.cuda.Device(0):
            # call directly without malloc.
            self.pool.free_all_blocks()
            self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_free_all_free(self):
        with cupy.cuda.Device(0):
            mem = self.pool.malloc(1).mem
            self.assertIsInstance(mem, memory.BaseMemory)
            self.assertIsInstance(mem, memory.PooledMemory)
            self.assertEqual(self.pool.n_free_blocks(), 0)
            mem.free()
            self.assertEqual(self.pool.n_free_blocks(), 1)
            with testing.assert_warns(DeprecationWarning):
                self.pool.free_all_free()
            self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_free_all_free_without_malloc(self):
        with cupy.cuda.Device(0):
            # call directly without malloc.
            with testing.assert_warns(DeprecationWarning):
                self.pool.free_all_free()
            self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_n_free_blocks_without_malloc(self):
        with cupy.cuda.Device(0):
            # call directly without malloc/free_all_free.
            self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_used_bytes(self):
        with cupy.cuda.Device(0):
            self.assertEqual(0, self.pool.used_bytes())

    def test_free_bytes(self):
        with cupy.cuda.Device(0):
            self.assertEqual(0, self.pool.free_bytes())

    def test_total_bytes(self):
        with cupy.cuda.Device(0):
            self.assertEqual(0, self.pool.total_bytes())


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
        with cupy.cuda.Device(0):
            self.assertEqual(0, self.pool.used_bytes())
            arr = cupy.arange(128, dtype=cupy.int64)
            self.assertEqual(1024, arr.data.mem.size)
            self.assertEqual(1024, self.pool.used_bytes())

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
                self.assertEqual(arr.data.mem.size, new_pool.used_bytes())
                threading.Barrier(2)
            assert memory.get_allocator() == self.pool.malloc

        with cupy.cuda.Device(0):
            t = threading.Thread(target=thread_body, args=(self,))
            t.daemon = True
            t.start()
            threading.Barrier(2)
            assert memory.get_allocator() == self.pool.malloc
            arr = cupy.ones(256, dtype=cupy.int64)
            threading.Barrier(2)
            self.assertEqual(arr.data.mem.size, self.pool.used_bytes())
            threading.Barrier(2)
            t.join()

    def test_thread_local_valid(self):
        new_pool = memory.MemoryPool()
        arr = None
        with cupy.cuda.using_allocator(new_pool.malloc):
            arr = cupy.zeros(128, dtype=cupy.int64)
            arr += 1
        # Check that arr and the pool have not ben released
        self.assertEqual(arr.data.mem.size, new_pool.used_bytes())
        assert arr.sum() == 128

    def test_reuse_between_thread(self):
        def job(self):
            cupy.arange(16)
            self._error = False

        # Run in main thread.
        self._error = True
        job(self)
        self.assertFalse(self._error)

        # Run in sub thread.
        self._error = True
        with cupy.cuda.Device(0):
            t = threading.Thread(target=job, args=(self,))
            t.daemon = True
            t.start()
            t.join()
        self.assertFalse(self._error)


@testing.gpu
class TestAllocatorDisabled(unittest.TestCase):

    def setUp(self):
        self.pool = cupy.get_default_memory_pool()

    def tearDown(self):
        memory.set_allocator(self.pool.malloc)

    def _check_pool_not_used(self):
        used_bytes = self.pool.used_bytes()
        with cupy.cuda.Device(0):
            arr = cupy.arange(128, dtype=cupy.int64)
            self.assertEqual(0, self.pool.used_bytes() - used_bytes)
            del arr

    def test(self):
        memory.set_allocator()
        self._check_pool_not_used()

    def test_none(self):
        memory.set_allocator(None)
        self._check_pool_not_used()


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

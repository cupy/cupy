import ctypes
import unittest

import cupy.cuda
from cupy.cuda import memory
from cupy import testing


class MockMemory(memory.Memory):
    cur_ptr = 1

    def __init__(self, size):
        self.ptr = MockMemory.cur_ptr
        MockMemory.cur_ptr += size
        self.size = size
        self.device = None

    def __del__(self):
        self.ptr = 0
        pass


def mock_alloc(size):
    mem = MockMemory(size)
    return memory.MemoryPointer(mem, 0)

# -----------------------------------------------------------------------------
# Memory pointer


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
        self.unit = self.pool._allocation_unit_size

    def test_round_size(self):
        self.assertEqual(self.pool._round_size(self.unit - 1), self.unit)
        self.assertEqual(self.pool._round_size(self.unit), self.unit)
        self.assertEqual(self.pool._round_size(self.unit + 1), self.unit * 2)

    def test_bin_index_from_size(self):
        self.assertEqual(self.pool._bin_index_from_size(self.unit - 1), 0)
        self.assertEqual(self.pool._bin_index_from_size(self.unit), 0)
        self.assertEqual(self.pool._bin_index_from_size(self.unit + 1), 1)

    def test_split(self):
        mem = MockMemory(self.unit * 4)
        chunk = memory.Chunk(mem, 0, mem.size)
        head, tail = self.pool._split(chunk, self.unit * 2)
        self.assertEqual(head.ptr,    chunk.ptr)
        self.assertEqual(head.offset, 0)
        self.assertEqual(head.size,   self.unit * 2)
        self.assertEqual(head.prev,   None)
        self.assertEqual(head.next,   tail)
        self.assertEqual(tail.ptr,    chunk.ptr + self.unit * 2)
        self.assertEqual(tail.offset, self.unit * 2)
        self.assertEqual(tail.size,   self.unit * 2)
        self.assertEqual(tail.prev,   head)
        self.assertEqual(tail.next,   None)

        head_of_head, tail_of_head = self.pool._split(head, self.unit)
        self.assertEqual(head_of_head.ptr,    chunk.ptr)
        self.assertEqual(head_of_head.offset, 0)
        self.assertEqual(head_of_head.size,   self.unit)
        self.assertEqual(head_of_head.prev,   None)
        self.assertEqual(head_of_head.next,   tail_of_head)
        self.assertEqual(tail_of_head.ptr,    chunk.ptr + self.unit)
        self.assertEqual(tail_of_head.offset, self.unit)
        self.assertEqual(tail_of_head.size,   self.unit)
        self.assertEqual(tail_of_head.prev,   head_of_head)
        self.assertEqual(tail_of_head.next,   tail)

        head_of_tail, tail_of_tail = self.pool._split(tail, self.unit)
        self.assertEqual(head_of_tail.ptr,    chunk.ptr + self.unit * 2)
        self.assertEqual(head_of_tail.offset, self.unit * 2)
        self.assertEqual(head_of_tail.size,   self.unit)
        self.assertEqual(head_of_tail.prev,   head)
        self.assertEqual(head_of_tail.next,   tail_of_tail)
        self.assertEqual(tail_of_tail.ptr,    chunk.ptr + self.unit * 3)
        self.assertEqual(tail_of_tail.offset, self.unit * 3)
        self.assertEqual(tail_of_tail.size,   self.unit)
        self.assertEqual(tail_of_tail.prev,   head_of_tail)
        self.assertEqual(tail_of_tail.next,   None)

    def test_merge(self):
        mem = MockMemory(self.unit * 4)
        chunk = memory.Chunk(mem, 0, mem.size)

        head, tail = self.pool._split(chunk, self.unit * 2)
        head_of_head, tail_of_head = self.pool._split(head, self.unit)
        head_of_tail, tail_of_tail = self.pool._split(tail, self.unit)

        merged_head = self.pool._merge(head_of_head, tail_of_head)
        self.assertEqual(merged_head.ptr,    head.ptr)
        self.assertEqual(merged_head.offset, head.offset)
        self.assertEqual(merged_head.size,   head.size)
        self.assertEqual(merged_head.prev,   head.prev)
        self.assertEqual(merged_head.next,   head.next)

        merged_tail = self.pool._merge(head_of_tail, tail_of_tail)
        self.assertEqual(merged_tail.ptr,    tail.ptr)
        self.assertEqual(merged_tail.offset, tail.offset)
        self.assertEqual(merged_tail.size,   tail.size)
        self.assertEqual(merged_tail.prev,   tail.prev)
        self.assertEqual(merged_tail.next,   tail.next)

        merged = self.pool._merge(merged_head, merged_tail)
        self.assertEqual(merged.ptr,    chunk.ptr)
        self.assertEqual(merged.offset, chunk.offset)
        self.assertEqual(merged.size,   chunk.size)
        self.assertEqual(merged.prev,   chunk.prev)
        self.assertEqual(merged.next,   chunk.next)

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

    def test_free(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(self.unit * 4)
        self.assertEqual(ptr1, p2.ptr)

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

        # do not free splitted blocks
        head = self.pool.malloc(self.unit * 2)
        tail = self.pool.malloc(self.unit * 2)
        tailptr = tail.ptr
        del tail
        self.pool.free_all_blocks()
        p = self.pool.malloc(self.unit * 2)
        self.assertEqual(tailptr, p.ptr)
        del head

    def test_free_all_free(self):
        p1 = self.pool.malloc(self.unit * 4)
        ptr1 = p1.ptr
        del p1
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
        del p3


@testing.gpu
class TestMemoryPool(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryPool()

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
            self.assertIsInstance(mem, memory.Memory)
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
            self.assertIsInstance(mem, memory.Memory)
            self.assertIsInstance(mem, memory.PooledMemory)
            self.assertEqual(self.pool.n_free_blocks(), 0)
            mem.free()
            self.assertEqual(self.pool.n_free_blocks(), 1)
            self.pool.free_all_free()
            self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_free_all_free_without_malloc(self):
        with cupy.cuda.Device(0):
            # call directly without malloc.
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

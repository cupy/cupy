import unittest

from cupy.cuda import pinned_memory
from cupy import testing


class MockMemory(pinned_memory.PinnedMemory):
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
    return pinned_memory.PinnedMemoryPointer(mem, 0)


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
        self.assertIsInstance(memptr2, pinned_memory.PinnedMemoryPointer)
        self.assertEqual(pval + 4, int(memptr2))

        memptr3 = 4 + memptr
        self.assertIsInstance(memptr3, pinned_memory.PinnedMemoryPointer)
        self.assertEqual(pval + 4, int(memptr3))

        memptr += 4
        self.assertIsInstance(memptr, pinned_memory.PinnedMemoryPointer)
        self.assertEqual(pval + 4, int(memptr))

    def test_sub(self):
        pval = MockMemory.cur_ptr
        memptr = mock_alloc(8) + 4

        memptr2 = memptr - 4
        self.assertIsInstance(memptr2, pinned_memory.PinnedMemoryPointer)
        self.assertEqual(pval, int(memptr2))

        memptr -= 4
        self.assertIsInstance(memptr, pinned_memory.PinnedMemoryPointer)
        self.assertEqual(pval, int(memptr))


# -----------------------------------------------------------------------------
# Memory pool


@testing.gpu
class TestSingleDeviceMemoryPool(unittest.TestCase):

    def setUp(self):
        self.pool = pinned_memory.PinnedMemoryPool(allocator=mock_alloc)

    def test_alloc(self):
        p1 = self.pool.malloc(1000)
        p2 = self.pool.malloc(1000)
        p3 = self.pool.malloc(2000)
        self.assertNotEqual(p1.ptr, p2.ptr)
        self.assertNotEqual(p1.ptr, p3.ptr)
        self.assertNotEqual(p2.ptr, p3.ptr)

    def test_free(self):
        p1 = self.pool.malloc(1000)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(1000)
        self.assertEqual(ptr1, p2.ptr)

    def test_free_different_size(self):
        p1 = self.pool.malloc(1000)
        ptr1 = p1.ptr
        del p1
        p2 = self.pool.malloc(2000)
        self.assertNotEqual(ptr1, p2.ptr)

    def test_free_all_blocks(self):
        p1 = self.pool.malloc(1000)
        ptr1 = p1.ptr
        del p1
        self.pool.free_all_blocks()
        p2 = self.pool.malloc(1000)
        self.assertNotEqual(ptr1, p2.ptr)

    def test_free_all_blocks2(self):
        mem = self.pool.malloc(1).mem
        self.assertIsInstance(mem, pinned_memory.PinnedMemory)
        self.assertIsInstance(mem, pinned_memory.PooledPinnedMemory)
        self.assertEqual(self.pool.n_free_blocks(), 0)
        mem.free()
        self.assertEqual(self.pool.n_free_blocks(), 1)
        self.pool.free_all_blocks()
        self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_zero_size_alloc(self):
        mem = self.pool.malloc(0).mem
        self.assertIsInstance(mem, pinned_memory.PinnedMemory)
        self.assertNotIsInstance(mem, pinned_memory.PooledPinnedMemory)

    def test_double_free(self):
        mem = self.pool.malloc(1).mem
        mem.free()
        mem.free()

    def test_free_all_blocks_without_malloc(self):
        # call directly without malloc.
        self.pool.free_all_blocks()
        self.assertEqual(self.pool.n_free_blocks(), 0)

    def test_n_free_blocks_without_malloc(self):
        # call directly without malloc/free_all_blocks.
        self.assertEqual(self.pool.n_free_blocks(), 0)

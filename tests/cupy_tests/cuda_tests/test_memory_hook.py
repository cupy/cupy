import unittest

import cupy.cuda
from cupy.cuda import memory
from cupy.cuda import memory_hook
from cupy import testing


class SimpleMemoryHook(memory_hook.MemoryHook):
    name = 'SimpleMemoryHook'

    def __init__(self):
        self.alloc_preprocess_history = []
        self.alloc_postprocess_history = []
        self.malloc_preprocess_history = []
        self.malloc_postprocess_history = []
        self.free_preprocess_history = []
        self.free_postprocess_history = []

    def alloc_preprocess(self, device_id, mem_size):
        self.alloc_preprocess_history.append(
            (device_id, mem_size))

    def alloc_postprocess(self, device_id, mem_size, mem_ptr):
        self.alloc_postprocess_history.append(
            (device_id, mem_size, mem_ptr))

    def malloc_preprocess(self, device_id, size, mem_size):
        self.malloc_preprocess_history.append(
            (device_id, size, mem_size))

    def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
        self.malloc_postprocess_history.append(
            (device_id, size, mem_size, mem_ptr, pmem_id))

    def free_preprocess(self, device_id, mem_size, mem_ptr, pmem_id):
        self.free_preprocess_history.append(
            (device_id, mem_size, mem_ptr, pmem_id))

    def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
        self.free_postprocess_history.append(
            (device_id, mem_size, mem_ptr, pmem_id))


@testing.gpu
class TestMemoryHook(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryPool()
        self.unit = 512

    def test_hook(self):
        hook = SimpleMemoryHook()
        with cupy.cuda.Device(0):
            with hook:
                mem = self.pool.malloc(1)
                ptr1, pmem1 = (mem.ptr, id(mem.mem))
                del mem
                mem = self.pool.malloc(1)
                ptr2, pmem2 = (mem.ptr, id(mem.mem))
                del mem
        self.assertEqual(1, len(hook.alloc_preprocess_history))
        self.assertEqual(1, len(hook.alloc_postprocess_history))
        self.assertEqual(2, len(hook.malloc_preprocess_history))
        self.assertEqual(2, len(hook.malloc_postprocess_history))
        self.assertEqual(2, len(hook.free_preprocess_history))
        self.assertEqual(2, len(hook.free_postprocess_history))
        self.assertEqual((0, self.unit),
                         hook.alloc_preprocess_history[0])
        self.assertEqual((0, self.unit, ptr1),
                         hook.alloc_postprocess_history[0])
        self.assertEqual((0, 1, self.unit),
                         hook.malloc_preprocess_history[0])
        self.assertEqual((0, 1, self.unit, ptr1, pmem1),
                         hook.malloc_postprocess_history[0])
        self.assertEqual((0, 1, self.unit),
                         hook.malloc_preprocess_history[1])
        self.assertEqual((0, 1, self.unit, ptr2, pmem2),
                         hook.malloc_postprocess_history[1])
        self.assertEqual((0, self.unit, ptr1, pmem1),
                         hook.free_preprocess_history[0])
        self.assertEqual((0, self.unit, ptr1, pmem1),
                         hook.free_postprocess_history[0])
        self.assertEqual((0, self.unit, ptr2, pmem2),
                         hook.free_preprocess_history[1])
        self.assertEqual((0, self.unit, ptr2, pmem2),
                         hook.free_postprocess_history[1])

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

    def alloc_preprocess(self, device_id, rounded_size):
        self.alloc_preprocess_history.append(
            (device_id, rounded_size))

    def alloc_postprocess(self, device_id, rounded_size, ptr):
        self.alloc_postprocess_history.append(
            (device_id, rounded_size, ptr))

    def malloc_preprocess(self, device_id, size, rounded_size):
        self.malloc_preprocess_history.append(
            (device_id, size, rounded_size))

    def malloc_postprocess(self, device_id, size, rounded_size, ptr):
        self.malloc_postprocess_history.append(
            (device_id, size, rounded_size, ptr))

    def free_preprocess(self, device_id, ptr, size):
        self.free_preprocess_history.append(
            (device_id, ptr, size))

    def free_postprocess(self, device_id, ptr, size):
        self.free_postprocess_history.append(
            (device_id, ptr, size))


@testing.gpu
class TestMemoryHook(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryPool()
        self.unit = 512

    def test_hook(self):
        hook = SimpleMemoryHook()
        with cupy.cuda.Device(0):
            with hook:
                ptr1 = self.pool.malloc(1).mem.ptr
                ptr2 = self.pool.malloc(1).mem.ptr
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
        self.assertEqual((0, 1, self.unit, ptr1),
                         hook.malloc_postprocess_history[0])
        self.assertEqual((0, 1, self.unit),
                         hook.malloc_preprocess_history[1])
        self.assertEqual((0, 1, self.unit, ptr2),
                         hook.malloc_postprocess_history[1])
        self.assertEqual((0, ptr1, self.unit),
                         hook.free_preprocess_history[0])
        self.assertEqual((0, ptr1, self.unit),
                         hook.free_postprocess_history[0])
        self.assertEqual((0, ptr2, self.unit),
                         hook.free_preprocess_history[1])
        self.assertEqual((0, ptr2, self.unit),
                         hook.free_postprocess_history[1])

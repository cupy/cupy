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

    def alloc_preprocess(self, device_id, rounded_size):
        self.alloc_preprocess_history.append(
            (device_id, rounded_size))

    def alloc_postprocess(self, device_id, rounded_size):
        self.alloc_postprocess_history.append(
            (device_id, rounded_size))

    def malloc_preprocess(self, device_id, size, rounded_size):
        self.malloc_preprocess_history.append(
            (device_id, size, rounded_size))

    def malloc_postprocess(self, device_id, size, rounded_size):
        self.malloc_postprocess_history.append(
            (device_id, size, rounded_size))


@testing.gpu
class TestMemoryHook(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryPool()
        self.unit = 512

    def test_hook(self):
        hook = SimpleMemoryHook()
        with cupy.cuda.Device(0):
            with hook:
                self.pool.malloc(1).mem
                self.pool.malloc(1).mem
        self.assertEqual(1, len(hook.alloc_preprocess_history))
        self.assertEqual(1, len(hook.alloc_postprocess_history))
        self.assertEqual(2, len(hook.malloc_preprocess_history))
        self.assertEqual(2, len(hook.malloc_postprocess_history))
        self.assertEqual((0, self.unit), hook.alloc_preprocess_history[0])
        self.assertEqual((0, self.unit), hook.alloc_postprocess_history[0])
        self.assertEqual((0, 1, self.unit), hook.malloc_preprocess_history[0])
        self.assertEqual((0, 1, self.unit), hook.malloc_postprocess_history[0])

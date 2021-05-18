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

    def alloc_preprocess(self, **kwargs):
        self.alloc_preprocess_history.append(kwargs)

    def alloc_postprocess(self, **kwargs):
        self.alloc_postprocess_history.append(kwargs)

    def malloc_preprocess(self, **kwargs):
        self.malloc_preprocess_history.append(kwargs)

    def malloc_postprocess(self, **kwargs):
        self.malloc_postprocess_history.append(kwargs)

    def free_preprocess(self, **kwargs):
        self.free_preprocess_history.append(kwargs)

    def free_postprocess(self, **kwargs):
        self.free_postprocess_history.append(kwargs)


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
        assert 1 == len(hook.alloc_preprocess_history)
        assert 1 == len(hook.alloc_postprocess_history)
        assert 2 == len(hook.malloc_preprocess_history)
        assert 2 == len(hook.malloc_postprocess_history)
        assert 2 == len(hook.free_preprocess_history)
        assert 2 == len(hook.free_postprocess_history)
        assert {'device_id': 0,
                'mem_size': self.unit} == hook.alloc_preprocess_history[0]
        assert {'device_id': 0, 'mem_size': self.unit,
                'mem_ptr': ptr1} == hook.alloc_postprocess_history[0]
        assert {'device_id': 0, 'size': 1,
                'mem_size': self.unit} == hook.malloc_preprocess_history[0]
        assert {'device_id': 0, 'size': 1, 'mem_size': self.unit,
                'mem_ptr': ptr1, 'pmem_id': pmem1
                } == hook.malloc_postprocess_history[0]
        assert {'device_id': 0, 'size': 1,
                'mem_size': self.unit} == hook.malloc_preprocess_history[1]
        assert {'device_id': 0, 'size': 1, 'mem_size': self.unit,
                'mem_ptr': ptr2, 'pmem_id': pmem2
                } == hook.malloc_postprocess_history[1]
        assert {'device_id': 0, 'mem_size': self.unit,
                'mem_ptr': ptr1, 'pmem_id': pmem1
                } == hook.free_preprocess_history[0]
        assert {'device_id': 0, 'mem_size': self.unit,
                'mem_ptr': ptr1, 'pmem_id': pmem1
                } == hook.free_postprocess_history[0]
        assert {'device_id': 0, 'mem_size': self.unit,
                'mem_ptr': ptr2, 'pmem_id': pmem2
                } == hook.free_preprocess_history[1]
        assert {'device_id': 0, 'mem_size': self.unit,
                'mem_ptr': ptr2, 'pmem_id': pmem2
                } == hook.free_postprocess_history[1]

import json
import six
import unittest

import cupy.cuda
from cupy.cuda import memory
from cupy.cuda import memory_hooks
from cupy import testing


@testing.gpu
class TestDebugPrintHook(unittest.TestCase):

    def setUp(self):
        self.io = six.StringIO()
        self.hook = memory_hooks.DebugPrintHook(file=self.io)
        self.pool = memory.MemoryPool()

    def test_print(self):
        device_id = 0
        size = 1
        unit = 512
        with cupy.cuda.Device(device_id):
            with self.hook:
                ptr1 = self.pool.malloc(size).mem.ptr
                ptr2 = self.pool.malloc(size).mem.ptr
        actual_lines = self.io.getvalue().splitlines()

        expect = {'hook': 'alloc', 'device_id': device_id,
                  'mem_size': unit, 'mem_ptr': ptr1}
        self.assertEqual(expect, json.loads(actual_lines[0]))

        expect = {'hook': 'malloc', 'device_id': device_id,
                  'size': size, 'mem_size': unit, 'mem_ptr': ptr1}
        self.assertEqual(expect, json.loads(actual_lines[1]))

        expect = {'hook': 'free', 'device_id': device_id,
                  'mem_size': unit, 'mem_ptr': ptr1}
        self.assertEqual(expect, json.loads(actual_lines[2]))

        expect = {'hook': 'malloc', 'device_id': device_id,
                  'size': size, 'mem_size': unit, 'mem_ptr': ptr2}
        self.assertEqual(expect, json.loads(actual_lines[3]))

        expect = {'hook': 'free', 'device_id': device_id,
                  'mem_size': unit, 'mem_ptr': ptr2}
        self.assertEqual(expect, json.loads(actual_lines[4]))

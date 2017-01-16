# coding: utf-8

import unittest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
class TestScan(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_scan(self, dtype):
        element_num = 10000

        if dtype in {cupy.int8, cupy.uint8, cupy.float16}:
            element_num = 100

        a = cupy.ones((element_num,), dtype=dtype)
        prefix_sum = cupy.core.core.scan(a)
        expect = cupy.arange(start=1, stop=element_num + 1).astype(dtype)

        testing.assert_array_equal(prefix_sum, expect)

    def test_check_1d_array(self):
        with self.assertRaises(TypeError):
            a = cupy.zeros((2, 2))
            cupy.core.core.scan(a)

    @testing.multi_gpu(2)
    def test_multi_gpu(self):
        with cuda.Device(0):
            a = cupy.zeros((10,))
            cupy.core.core.scan(a)
        with cuda.Device(1):
            a = cupy.zeros((10,))
            cupy.core.core.scan(a)

    @testing.for_all_dtypes()
    def test_scan_out(self, dtype):
        element_num = 10000

        if dtype in {cupy.int8, cupy.uint8, cupy.float16}:
            element_num = 100

        a = cupy.ones((element_num,), dtype=dtype)
        b = cupy.zeros_like(a)
        cupy.core.core.scan(a, b)
        expect = cupy.arange(start=1, stop=element_num + 1).astype(dtype)

        testing.assert_array_equal(b, expect)

        cupy.core.core.scan(a, a)
        testing.assert_array_equal(a, expect)

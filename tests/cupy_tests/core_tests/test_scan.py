# coding: utf-8

import unittest

import cupy
from cupy import testing


class TestScan(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_scan(self, dtype):
        element_num = 10000

        if dtype in [cupy.int8, cupy.uint8]:
            element_num = 100

        a = cupy.ones((element_num,), dtype=dtype)
        prefix_sum = cupy.core.core.scan(a)
        expect = cupy.arange(start=1, stop=element_num + 1).astype(dtype)

        testing.assert_array_equal(prefix_sum, expect)

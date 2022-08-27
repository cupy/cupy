import unittest

import cupy
from cupy import testing


@testing.gpu
class TestByteBounds(unittest.TestCase):

    def test_contiguous_1d_arrays(self):
        a = cupy.arange(12, dtype=cupy.float32)
        a_low = a.data.ptr
        a_high = a_low + 12*4
        return cupy.byte_bounds(a) == (a_low, a_high)

    def test_contiguous_2d_arrays(self):
        a = cupy.zeros((4, 7), dtype=cupy.float32)
        a_low = a.data.ptr
        a_high = a_low + 4*7*4
        return cupy.byte_bounds(a) == (a_low, a_high)

    def test_non_contiguous_1d_arrays(self):
        a = cupy.arange(12, dtype=cupy.float32)
        a = a[::2]
        a_low = a.data.ptr
        a_high = a_low + 11*4  # a[10]
        return cupy.byte_bounds(a) == (a_low, a_high)

    def test_non_contiguous_2d_arrays(self):
        a = cupy.zeros((4, 7), dtype=cupy.float32)
        a = a[::2, ::2]
        a_low = a.data.ptr
        a_high = a_low + 3*7*4  # a[2][6]
        return cupy.byte_bounds(a) == (a_low, a_high)

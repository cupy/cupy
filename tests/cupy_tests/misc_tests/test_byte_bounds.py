import numpy
import unittest

from cupy import testing


@testing.gpu
class TestByteBounds(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_contiguous_1d_arrays(self, xp):
        a = xp.arange(12, dtype=xp.float32)
        if xp == numpy:
            a_low, a_high = xp.byte_bounds(a)
        else:
            a_low = a.data.ptr
            a_high = a_low + 12*4
        return xp.byte_bounds(a) == (a_low, a_high)

    @testing.numpy_cupy_equal()
    def test_contiguous_2d_arrays(self, xp):
        a = xp.zeros((4, 7), dtype=xp.float32)
        if xp == numpy:
            a_low, a_high = xp.byte_bounds(a)
        else:
            a_low = a.data.ptr
            a_high = a_low + 4*7*4
        return xp.byte_bounds(a) == (a_low, a_high)

    @testing.numpy_cupy_equal()
    def test_non_contiguous_1d_arrays(self, xp):
        a = xp.arange(12, dtype=xp.float32)
        a = a[::2]
        if xp == numpy:
            a_low, a_high = xp.byte_bounds(a)
        else:
            a_low = a.data.ptr
            a_high = a_low + 11*4  # a[10]
        return xp.byte_bounds(a) == (a_low, a_high)

    @testing.numpy_cupy_equal()
    def test_non_contiguous_2d_arrays(self, xp):
        a = xp.zeros((4, 7), dtype=xp.float32)
        a = a[::2, ::2]
        if xp == numpy:
            a_low, a_high = xp.byte_bounds(a)
        else:
            a_low = a.data.ptr
            a_high = a_low + 3*7*4  # a[2][6]
        return xp.byte_bounds(a) == (a_low, a_high)

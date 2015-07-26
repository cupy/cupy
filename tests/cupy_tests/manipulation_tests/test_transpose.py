import unittest

import cupy
from cupy import testing


@testing.gpu
class TestTranspose(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_rollaxis(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        return xpy.rollaxis(a, 2)

    def test_rollaxis_failure(self):
        a = testing.shaped_arange((2, 3, 4))
        with self.assertRaises(ValueError):
            cupy.rollaxis(a, 3)

    @testing.numpy_cupy_array_equal()
    def test_swapaxes(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        return xpy.swapaxes(a, 2, 0)

    def test_swapaxes_failure(self):
        a = testing.shaped_arange((2, 3, 4))
        with self.assertRaises(ValueError):
            cupy.swapaxes(a, 3, 0)

    @testing.numpy_cupy_array_equal()
    def test_transpose(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        return a.transpose(-1, 0, 1)

    @testing.numpy_cupy_array_equal()
    def test_transpose_empty(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        return a.transpose()

    @testing.numpy_cupy_array_equal()
    def test_external_transpose(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        return xpy.transpose(a)

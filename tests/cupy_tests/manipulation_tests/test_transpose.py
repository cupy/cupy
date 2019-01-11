import unittest

import cupy
from cupy import testing


@testing.gpu
class TestTranspose(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 2])

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, 1, -1)

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis3(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 2], [1, 0])

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis4(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [2, 0], [1, 0])

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis5(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [2, 0], [0, 1])

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis6(self, xp):
        a = testing.shaped_arange((2, 3, 4, 5, 6), xp)
        return xp.moveaxis(a, [0, 2, 1], [3, 4, 0])

    # dim is too large
    @testing.numpy_cupy_raises()
    @testing.with_requires('numpy>=1.13')
    def test_moveaxis_invalid1_1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 3])

    def test_moveaxis_invalid1_2(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        with self.assertRaises(cupy.core._AxisError):
            return cupy.moveaxis(a, [0, 1], [1, 3])

    # dim is too small
    @testing.numpy_cupy_raises()
    @testing.with_requires('numpy>=1.13')
    def test_moveaxis_invalid2_1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, -4], [1, 2])

    def test_moveaxis_invalid2_2(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        with self.assertRaises(cupy.core._AxisError):
            return cupy.moveaxis(a, [0, -4], [1, 2])

    # len(source) != len(destination)
    @testing.numpy_cupy_raises()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis_invalid3(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1, 2], [1, 2])

    # len(source) != len(destination)
    @testing.numpy_cupy_raises()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis_invalid4(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 2, 0])

    # Use the same axis twice
    def test_moveaxis_invalid5_1(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        with self.assertRaises(cupy.core._AxisError):
            return cupy.moveaxis(a, [1, -1], [1, 3])

    def test_moveaxis_invalid5_2(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        with self.assertRaises(cupy.core._AxisError):
            return cupy.moveaxis(a, [0, 1], [-1, 2])

    def test_moveaxis_invalid5_3(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        with self.assertRaises(cupy.core._AxisError):
            return cupy.moveaxis(a, [0, 1], [1, 1])

    @testing.numpy_cupy_array_equal()
    def test_rollaxis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.rollaxis(a, 2)

    def test_rollaxis_failure(self):
        a = testing.shaped_arange((2, 3, 4))
        with self.assertRaises(ValueError):
            cupy.rollaxis(a, 3)

    @testing.numpy_cupy_array_equal()
    def test_swapaxes(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.swapaxes(a, 2, 0)

    def test_swapaxes_failure(self):
        a = testing.shaped_arange((2, 3, 4))
        with self.assertRaises(ValueError):
            cupy.swapaxes(a, 3, 0)

    @testing.numpy_cupy_array_equal()
    def test_transpose(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose(-1, 0, 1)

    @testing.numpy_cupy_array_equal()
    def test_transpose_empty(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose()

    @testing.numpy_cupy_array_equal()
    def test_transpose_none(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose(None)

    @testing.numpy_cupy_array_equal()
    def test_external_transpose(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.transpose(a, (-1, 0, 1))

    @testing.numpy_cupy_array_equal()
    def test_external_transpose_all(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.transpose(a)

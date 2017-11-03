import unittest

import cupy
from cupy import testing


@testing.gpu
class TestTranspose(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 2])

    # dim is too large
    @testing.numpy_cupy_raises()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis_invalid1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 3])

    # dim is too small
    @testing.numpy_cupy_raises()
    @testing.with_requires('numpy>=1.11')
    def test_moveaxis_invalid2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, -3], [1, 2])

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

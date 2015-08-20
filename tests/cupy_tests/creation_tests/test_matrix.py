import unittest

from cupy import testing


@testing.gpu
class TestMatrix(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_diag1(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diag(a)

    @testing.numpy_cupy_array_equal()
    def test_diag2(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diag(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_diag3(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diag(a, -2)

    @testing.numpy_cupy_array_equal()
    def test_diagflat1(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diagflat(a)

    @testing.numpy_cupy_array_equal()
    def test_diagflat2(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diagflat(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_diagflat3(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diagflat(a, -2)

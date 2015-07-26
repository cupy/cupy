import unittest

from cupy import testing


@testing.gpu
class TestMatrix(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_diag1(self, xpy):
        a = testing.shaped_arange((3, 3), xpy)
        return xpy.diag(a)

    @testing.numpy_cupy_array_equal()
    def test_diag2(self, xpy):
        a = testing.shaped_arange((3, 3), xpy)
        return xpy.diag(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_diag3(self, xpy):
        a = testing.shaped_arange((3, 3), xpy)
        return xpy.diag(a, -2)

    @testing.numpy_cupy_array_equal()
    def test_diagflat1(self, xpy):
        a = testing.shaped_arange((3, 3), xpy)
        return xpy.diagflat(a)

    @testing.numpy_cupy_array_equal()
    def test_diagflat2(self, xpy):
        a = testing.shaped_arange((3, 3), xpy)
        return xpy.diagflat(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_diagflat3(self, xpy):
        a = testing.shaped_arange((3, 3), xpy)
        return xpy.diagflat(a, -2)

import unittest

from cupy import testing


@testing.gpu
class TestIndexing(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_take_by_scalar(self, xpy):
        a = testing.shaped_arange((2, 4, 3), xpy)
        return a.take(2, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_take_by_array(self, xpy):
        a = testing.shaped_arange((2, 4, 3), xpy)
        b = xpy.array([[1, 3], [2, 0]])
        return a.take(b, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_take_no_axis(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        b = xpy.array([[10, 5], [3, 20]])
        return a.take(b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal1(self, xpy, dtype):
        a = testing.shaped_arange((3, 4, 5), xpy, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal2(self, xpy, dtype):
        a = testing.shaped_arange((3, 4, 5), xpy, dtype)
        return a.diagonal(-1, 2, 0)

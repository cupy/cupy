import unittest

from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty(self, xpy, dtype):
        a = xpy.empty((2, 3, 4), dtype=dtype)
        a.fill(0)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = xpy.empty_like(a)
        b.fill(0)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_eye(self, xpy, dtype):
        return xpy.eye(5, 4, 1, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_identity(self, xpy, dtype):
        return xpy.identity(4, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros(self, xpy, dtype):
        return xpy.zeros((2, 3, 4), dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_like(self, xpy, dtype):
        a = xpy.ndarray((2, 3, 4), dtype=dtype)
        return xpy.zeros_like(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones(self, xpy, dtype):
        return xpy.ones((2, 3, 4), dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones_like(self, xpy, dtype):
        a = xpy.ndarray((2, 3, 4), dtype=dtype)
        return xpy.ones_like(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full(self, xpy, dtype):
        return xpy.full((2, 3, 4), 1, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_like(self, xpy, dtype):
        a = xpy.ndarray((2, 3, 4), dtype=dtype)
        return xpy.full_like(a, 1)

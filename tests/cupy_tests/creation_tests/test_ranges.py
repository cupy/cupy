import unittest

from cupy import testing


@testing.gpu
class TestRanges(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange(self, xpy, dtype):
        return xpy.arange(10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange2(self, xpy, dtype):
        return xpy.arange(5, 10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange3(self, xpy, dtype):
        return xpy.arange(1, 11, 2, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange4(self, xpy, dtype):
        return xpy.arange(20, 2, -3, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace(self, xpy, dtype):
        return xpy.linspace(0, 10, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace2(self, xpy, dtype):
        return xpy.linspace(10, 0, 5, dtype=dtype)

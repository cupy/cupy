import unittest

from cupy import testing


@testing.gpu
class TestRanges(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange(self, xp, dtype):
        return xp.arange(10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange2(self, xp, dtype):
        return xp.arange(5, 10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange3(self, xp, dtype):
        return xp.arange(1, 11, 2, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange4(self, xp, dtype):
        return xp.arange(20, 2, -3, dtype=dtype)

    @testing.numpy_cupy_array_equal()
    def test_arange_no_dtype_int(self, xp):
        return xp.arange(1, 11, 2)

    @testing.numpy_cupy_array_equal()
    def test_arange_no_dtype_float(self, xp):
        return xp.arange(1.0, 11.0, 2.0)

    @testing.numpy_cupy_array_equal()
    def test_arange_negative_size(self, xp):
        return xp.arange(3, 1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace2(self, xp, dtype):
        return xp.linspace(10, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_zero_num(self, xp, dtype):
        return xp.linspace(0, 10, 0, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_one_num(self, xp, dtype):
        return xp.linspace(0, 10, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_no_endpoint(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype, endpoint=False)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 5, dtype=dtype, retstep=True)
        self.assertEqual(step, 2.5)
        return x

    @testing.numpy_cupy_allclose()
    def test_linspace_no_dtype_int(self, xp):
        return xp.linspace(0, 10)

    @testing.numpy_cupy_array_equal()
    def test_linspace_no_dtype_float(self, xp):
        return xp.arange(0.0, 10.0)

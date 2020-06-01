import unittest

import pytest

import numpy
import cupy
from cupy import testing


@testing.gpu
class TestUnique(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_index(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_index=True)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_inverse(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_inverse=True)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_counts(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_counts=True)[1]


@testing.parameterize(*testing.product({
    'trim': ['fb', 'f', 'b']
}))
@testing.gpu
class TestTrim_zeros(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_non_zeros(self, xp, dtype):
        a = xp.array([-1, 2, -3, 7], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_trimmed(self, xp, dtype):
        a = xp.array([1, 0, 2, 3, 0, 5], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_all_zeros(self, xp, dtype):
        a = xp.zeros(shape=(1000,), dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_front_zeros(self, xp, dtype):
        a = xp.array([0, 0, 4, 1, 0, 2, 3, 0, 5], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_back_zeros(self, xp, dtype):
        a = xp.array([1, 0, 2, 3, 0, 5, 0, 0, 0], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    def test_trim_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            with pytest.raises(TypeError):
                xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    def test_trim_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3), xp, dtype=dtype)
            with pytest.raises(ValueError):
                xp.trim_zeros(a, trim=self.trim)

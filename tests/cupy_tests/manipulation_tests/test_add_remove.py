import unittest

import pytest

import numpy
import cupy
from cupy import testing


@testing.gpu
class TestAppend(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test(self, xp, dtype1, dtype2):
        a = testing.shaped_random((3, 4, 5), xp, dtype1)
        b = testing.shaped_random((6, 7), xp, dtype2)
        return xp.append(a, b)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_scalar_lhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype1).type(10).item()
        return xp.append(scalar, xp.arange(20, dtype=dtype2))

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_scalar_rhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype2).type(10).item()
        return xp.append(xp.arange(20, dtype=dtype1), scalar)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_numpy_scalar_lhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype1).type(10)
        return xp.append(scalar, xp.arange(20, dtype=dtype2))

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_numpy_scalar_rhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype2).type(10)
        return xp.append(xp.arange(20, dtype=dtype1), scalar)

    @testing.numpy_cupy_array_equal()
    def test_scalar_both(self, xp):
        return xp.append(10, 10)

    @testing.numpy_cupy_array_equal()
    def test_axis(self, xp):
        a = testing.shaped_random((3, 4, 5), xp, xp.float32)
        b = testing.shaped_random((3, 10, 5), xp, xp.float32)
        return xp.append(a, b, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_zerodim(self, xp):
        return xp.append(xp.array(0), xp.arange(10))

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.append(xp.array([]), xp.arange(10))


@testing.gpu
class TestResize(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test(self, xp):
        return xp.resize(xp.arange(10), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_remainder(self, xp):
        return xp.resize(xp.arange(8), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_shape_int(self, xp):
        return xp.resize(xp.arange(10), 15)

    @testing.numpy_cupy_array_equal()
    def test_scalar(self, xp):
        return xp.resize(2, (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_scalar_shape_int(self, xp):
        return xp.resize(2, 10)

    @testing.numpy_cupy_array_equal()
    def test_typed_scalar(self, xp):
        return xp.resize(xp.float32(10.0), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_zerodim(self, xp):
        return xp.resize(xp.array(0), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.resize(xp.array([]), (10, 10))


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

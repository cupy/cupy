import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestPolynomial(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyvander(self, xp, dtype):
        a = testing.shaped_random((3,), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 3)

    @testing.for_all_dtypes()
    def test_polyvander_negative_degree(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyvander(a, -3)

    @testing.for_all_dtypes()
    def test_polyvander_non_integral_float_degree(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polynomial.polynomial.polyvander(a, 2.6)

    @testing.for_all_dtypes()
    def test_polyvander_integral_float_degree(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, dtype)
            with pytest.raises(DeprecationWarning):
                xp.polynomial.polynomial.polyvander(a, 5.0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyvander_zeros(self, xp, dtype):
        a = xp.zeros(10, dtype)
        return xp.polynomial.polynomial.polyvander(a, 5)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyvander_ndim(self, xp, dtype):
        a = testing.shaped_random((3, 2, 1), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyvander_zero_dim(self, xp, dtype):
        a = testing.shaped_random((), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 5)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polycompanion(self, xp, dtype):
        a = testing.shaped_random((1000,), xp, dtype)
        return xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes()
    def test_polycompanion_zeros(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros(10, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes()
    def test_polycompanion_empty(self, dtype):
        for xp in (numpy, cupy):
            a = testing.empty(xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_polycompanion_trailing_zeros(self, xp, dtype):
        a = xp.array([3, 5, 7, 0, 0, 0], dtype)
        return xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes()
    def test_polycompanion_single_value1(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array([3, 0, 0, 0], dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_polycompanion_single_value2(self, xp, dtype):
        a = xp.array([0, 0, 0, 3], dtype)
        return xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_polycompanion_single_value3(self, xp, dtype):
        a = xp.array([0, 3, 0, 0], dtype)
        return xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes()
    def test_polycompanion_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((3, 4, 5), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polycompanion(a)

    @testing.for_all_dtypes()
    def test_polycompanion_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polycompanion(a)

    def test_polycompanion_nocommon_types(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((5,), xp, dtype=bool)
            with pytest.raises(Exception):
                xp.polynomial.polynomial.polycompanion(a)

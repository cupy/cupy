from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing


class TestPolynomial(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyvander1(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 20)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyvander2(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 10)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyvander3(self, xp, dtype):
        a = testing.shaped_random((100,), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 10)

    @testing.for_all_dtypes()
    def test_polyvander_negative_degree(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyvander(a, -3)

    @testing.with_requires('numpy>=1.17')
    @testing.for_all_dtypes()
    def test_polyvander_non_integral_float_degree(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polynomial.polynomial.polyvander(a, 2.6)

    @testing.with_requires('numpy>=2.0')
    @testing.for_all_dtypes(no_float16=True)
    def test_polyvander_integral_float_degree(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polynomial.polynomial.polyvander(a, 5.0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_polyvander_zeros(self, xp, dtype):
        a = xp.zeros(10, dtype)
        return xp.polynomial.polynomial.polyvander(a, 10)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyvander_ndim(self, xp, dtype):
        a = testing.shaped_random((3, 2, 1), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
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
    def test_polycompanion_zerosize(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros((0,), dtype)
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

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polymul_simple(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        b = xp.array([3, 2, 1], dtype=dtype)
        return xp.polynomial.polynomial.polymul(a, b)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polymul_zeros(self, xp, dtype):
        a = xp.array([0, 0, 1], dtype=dtype)
        b = xp.array([0, 1, 0], dtype=dtype)
        return xp.polynomial.polynomial.polymul(a, b)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-7)
    def test_polymul_monomials(self, xp, dtype):
        results = []
        for i in range(5):
            for j in range(5):
                a = xp.zeros(i + 1, dtype=dtype)
                b = xp.zeros(j + 1, dtype=dtype)
                a[-1] = 1
                b[-1] = 1
                results.append(xp.polynomial.polynomial.polymul(a, b))
        return results

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polymul_random(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        b = testing.shaped_random((5,), xp, dtype)
        return xp.polynomial.polynomial.polymul(a, b)

    def test_polymul_types(self):
        for dtype in [numpy.float32, numpy.float64, numpy.complex64,
                      numpy.complex128, numpy.int32, numpy.int64]:
            a_numpy = numpy.array([1, 2, 3], dtype=dtype)
            b_numpy = numpy.array([3, 2, 1], dtype=dtype)
            a_cupy = cupy.array([1, 2, 3], dtype=dtype)
            b_cupy = cupy.array([3, 2, 1], dtype=dtype)

            res_numpy = numpy.polynomial.polynomial.polymul(a_numpy, b_numpy)
            res_cupy = cupy.polynomial.polynomial.polymul(a_cupy, b_cupy)

            cupy.testing.assert_allclose(res_cupy, res_numpy, rtol=1e-5)

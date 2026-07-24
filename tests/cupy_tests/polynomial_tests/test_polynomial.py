from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing


class TestPolynomial(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_by_constant(self, xp, dtype):
        c1 = xp.array([1, 2, 3], dtype)
        c2 = xp.array([2], dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_same_degree(self, xp, dtype):
        c1 = xp.array([1, 2, 3], dtype)
        c2 = xp.array([3, 2, 1], dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_lower_degree_dividend(self, xp, dtype):
        c1 = xp.array([1, 2], dtype)
        c2 = xp.array([1, 2, 3], dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    def test_div_zero_divisor(self, dtype):
        for xp in (numpy, cupy):
            c1 = xp.array([1, 2, 3], dtype)
            c2 = xp.array([0], dtype)
            with pytest.raises(ZeroDivisionError):
                xp.polynomial.polynomial.polydiv(c1, c2)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_exact(self, xp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip("unsigned dtype: negative literals not representable")
        c1 = xp.array([1, -2, 1], dtype)
        c2 = xp.array([1, -1], dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_zero_dividend(self, xp, dtype):
        c1 = xp.array([0], dtype)
        c2 = xp.array([1, 2], dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_fractional(self, xp, dtype):
        if xp.dtype(dtype).kind != 'f':
            pytest.skip("float dtype only: fractional division test")
        c1 = xp.array([0.5, 1.5], dtype)
        tmp = xp.array([1, -1], dtype=xp.float64)
        c2 = tmp.astype(dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_div_trimmed_remainder(self, xp, dtype):
        c1 = xp.array([1., 2., 3.], dtype)
        c2 = xp.array([3., 2., 1.], dtype)
        _, r = xp.polynomial.polynomial.polydiv(c1, c2)
        if xp is cupy:
            return cupy.polynomial.polyutils.trimseq(r)
        else:
            return numpy.polynomial.polyutils.trimseq(r)

    @testing.for_all_dtypes()
    def test_div_boolean_input(self, dtype):
        for xp in (numpy, cupy):
            if xp.dtype(dtype).kind != 'b':
                pytest.skip("boolean dtype only")
            c1 = xp.array([True, True, True], dtype)
            c2 = xp.array([True], dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polydiv(c1, c2)

    @testing.for_all_dtypes()
    def test_div_empty_input(self, dtype):
        for xp in (numpy, cupy):
            c1 = xp.array([], dtype)
            c2 = xp.array([1, 2], dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polydiv(c1, c2)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    def test_div_non_1d_input(self, dtype):
        for xp in (numpy, cupy):
            c1 = xp.array([[1, 2], [3, 4]], dtype)
            c2 = xp.array([1, 2], dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polydiv(c1, c2)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    def test_unsigned_negative_literal_creation(self, dtype):
        for xp in (numpy, cupy):
            if xp.dtype(dtype).kind != 'u':
                pytest.skip(
                    "unsigned dtype: negative literals not representable")
            with pytest.raises(OverflowError):
                xp.array([1, -1], dtype)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-8)
    def test_div_complex(self, xp, dtype):
        dt = xp.dtype(dtype)
        if dt.kind != 'c':
            pytest.skip("skip: only complex dtype supported")
        c1 = xp.array([1+2j, -3+0.5j, 2-1j], dtype)
        c2 = xp.array([1-1j, 2j], dtype)
        q, r = xp.polynomial.polynomial.polydiv(c1, c2)
        return xp.concatenate((q, r))

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

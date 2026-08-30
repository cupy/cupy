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
    ##

    def test_polyint_raises_TypeError(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.polynomial.polynomial.polyint([0], .5)

    def test_polyint_raises_ValueError_negative_m(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyint([0], -1)

    def test_polyint_raises_ValueError_too_many_constants(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyint([0], 1, [0, 0])

    def test_polyint_raises_ValueError_lbnd(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyint([0], lbnd=[0])

    def test_polyint_raises_ValueError_scl(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyint([0], scl=[0])

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_zeroth(self, xp, dtype):
        c = testing.shaped_random((5,), xp, dtype)
        return xp.polynomial.polynomial.polyint(c, m=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_simple(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        return xp.polynomial.polynomial.polyint(c, m=1, k=[5])

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_with_lbnd(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        result = xp.polynomial.polynomial.polyint(c, m=1, k=[5], lbnd=-1)
        p = xp.polynomial.polynomial.polyval(-1, result)
        return xp.concatenate([result, xp.array([p], dtype=result.dtype)])

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_with_scaling(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        return xp.polynomial.polynomial.polyint(c, m=1, k=[5], scl=2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_multiple(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        return xp.polynomial.polynomial.polyint(c, m=3)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_multiple_with_k(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        k = xp.array([1, 2, 3])
        return xp.polynomial.polynomial.polyint(c, m=3, k=k)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_multiple_with_lbnd(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        return xp.polynomial.polynomial.polyint(c, m=3, lbnd=-1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_multiple_with_scaling(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        return xp.polynomial.polynomial.polyint(c, m=3, scl=2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_axis0(self, xp, dtype):
        c = testing.shaped_random((3, 4), xp, dtype)
        return xp.polynomial.polynomial.polyint(c, axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_axis1(self, xp, dtype):
        c = testing.shaped_random((3, 4), xp, dtype)
        return xp.polynomial.polynomial.polyint(c, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyint_deriv_inverse_relation(self, xp, dtype):
        c = testing.shaped_random((5,), xp, dtype)
        integ = xp.polynomial.polynomial.polyint(c, m=2)
        deriv = xp.polynomial.polynomial.polyder(integ, m=2)
        return deriv

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_zero_deriv(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.polynomial.polynomial.polyder(a, m=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_simple(self, xp, dtype):
        c = xp.array([1, 2, 3], dtype=dtype)
        return xp.polynomial.polynomial.polyder(c)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_inverse_polyint_m1(self, xp, dtype):
        c = testing.shaped_random((5,), xp, dtype)
        polyint_c = xp.polynomial.polynomial.polyint(c, m=1)
        return xp.polynomial.polynomial.polyder(polyint_c, m=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_inverse_polyint_m2(self, xp, dtype):
        c = testing.shaped_random((5,), xp, dtype)
        polyint_c = xp.polynomial.polynomial.polyint(c, m=2)
        return xp.polynomial.polynomial.polyder(polyint_c, m=2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_with_scl_m1(self, xp, dtype):
        c = testing.shaped_random((5,), xp, dtype)
        polyint_c = xp.polynomial.polynomial.polyint(c, m=1, scl=2)
        return xp.polynomial.polynomial.polyder(polyint_c, m=1, scl=0.5)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_with_scl_m2(self, xp, dtype):
        c = testing.shaped_random((5,), xp, dtype)
        polyint_c = xp.polynomial.polynomial.polyint(c, m=2, scl=2)
        return xp.polynomial.polynomial.polyder(polyint_c, m=2, scl=0.5)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_axis0(self, xp, dtype):
        c = testing.shaped_random((3, 4), xp, dtype)
        return xp.polynomial.polynomial.polyder(c, axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyder_axis1(self, xp, dtype):
        c = testing.shaped_random((3, 4), xp, dtype)
        return xp.polynomial.polynomial.polyder(c, axis=1)

    def test_polyder_invalid_params(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.polynomial.polynomial.polyder([0], .5)

            with pytest.raises(ValueError):
                xp.polynomial.polynomial.polyder([0], -1)

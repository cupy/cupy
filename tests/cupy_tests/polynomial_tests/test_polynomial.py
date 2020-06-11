import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestPolynomial(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
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

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_polyvander_zeros(self, xp, dtype):
        a = xp.zeros(10, dtype)
        return xp.polynomial.polynomial.polyvander(a, 5)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_polyvander_ndim(self, xp, dtype):
        a = testing.shaped_random((3, 2, 1), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_polyvander_zero_dim(self, xp, dtype):
        a = testing.shaped_random((), xp, dtype)
        return xp.polynomial.polynomial.polyvander(a, 5)

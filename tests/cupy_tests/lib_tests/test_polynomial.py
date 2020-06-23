import unittest

import pytest
import numpy

import cupy
from cupy import testing


@testing.gpu
class TestPoly1d(unittest.TestCase):

    def make_poly1d(self, xp, dtype):
        a1 = testing.shaped_arange((4,), xp, dtype)
        a2 = xp.zeros((4,), dtype)
        b1 = xp.poly1d(a1)
        b2 = xp.poly1d(a2)
        return b1, b2

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.poly1d(a).coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_leading_zeros(self, xp, dtype):
        a = xp.array([0, 0, 1, 2, 3], dtype)
        return xp.poly1d(a).coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_variable(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.poly1d(a, variable='p').variable

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_poly1d_astype(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((5,), xp, dtype1)
        return xp.poly1d(a).__array__(dtype2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_order(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return xp.poly1d(a).order

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_eq(self, xp, dtype):
        a, b = self.make_poly1d(xp, dtype)
        return a == b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_neg(self, xp, dtype):
        a, b = self.make_poly1d(xp, dtype)
        return a != b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem1(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return xp.poly1d(a)[-1]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem2(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return xp.poly1d(a)[5]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem3(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return xp.poly1d(a)[100]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_setitem(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        b = xp.poly1d(a)
        b[100] = 20
        return b[100]

    @testing.for_all_dtypes()
    def test_poly1d_setitem_neg(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((10,), xp, dtype)
            b = xp.poly1d(a)
            with pytest.raises(ValueError):
                b[-1] = 20

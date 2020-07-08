import unittest

import pytest
import numpy

import cupy
import cupyx
from cupy import testing


@testing.parameterize(
    {'variable': None},
    {'variable': 'y'},
)
@testing.gpu
class TestPoly1dInit(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_poly1d_numpy_array(self, xp, dtype):
        a = numpy.arange(5, dtype=dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_cupy_array(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs

    @testing.numpy_cupy_array_equal()
    def test_poly1d_list(self, xp):
        with cupyx.allow_synchronize(False):
            out = xp.poly1d([1, 2, 3, 4], variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_numpy_poly1d(self, xp, dtype):
        array = testing.shaped_arange((5,), numpy, dtype)
        a = numpy.poly1d(array)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_numpy_poly1d_variable(self, xp, dtype):
        array = testing.shaped_arange((5,), numpy, dtype)
        a = numpy.poly1d(array, variable='z')
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'z')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_cupy_poly1d(self, xp, dtype):
        array = testing.shaped_arange((5,), xp, dtype)
        a = xp.poly1d(array)
        out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_cupy_poly1d_variable(self, xp, dtype):
        array = testing.shaped_arange((5,), xp, dtype)
        a = xp.poly1d(array, variable='z')
        out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'z')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_zero_dim(self, xp, dtype):
        a = testing.shaped_arange((), xp, dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_zero_size(self, xp, dtype):
        a = testing.shaped_arange((0,), xp, dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out.coeffs


@testing.gpu
class TestPoly1d(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_leading_zeros(self, xp, dtype):
        a = xp.array([0, 0, 1, 2, 3], dtype)
        return xp.poly1d(a).coeffs

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_poly1d_neg(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = -xp.poly1d(a)
        return b.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_order(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return xp.poly1d(a).order

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_order_leading_zeros(self, xp, dtype):
        a = xp.array([0, 0, 1, 2, 3, 0], dtype)
        return xp.poly1d(a).order

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem1(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        with cupyx.allow_synchronize(False):
            return xp.poly1d(a)[-1]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem2(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        with cupyx.allow_synchronize(False):
            return xp.poly1d(a)[5]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem3(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        with cupyx.allow_synchronize(False):
            return xp.poly1d(a)[100]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_getitem4(self, xp, dtype):
        a = xp.array([0, 0, 1, 2, 3, 0], dtype)
        with cupyx.allow_synchronize(False):
            return xp.poly1d(a)[2]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_setitem(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        b = xp.poly1d(a)
        with cupyx.allow_synchronize(False):
            b[100] = 20
        return b.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_setitem_leading_zeros(self, xp, dtype):
        a = xp.array([0, 0, 0, 2, 3, 0], dtype)
        b = xp.poly1d(a)
        with cupyx.allow_synchronize(False):
            b[1] = 10
        return b.coeffs

    @testing.for_all_dtypes()
    def test_poly1d_setitem_neg(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((10,), xp, dtype)
            b = xp.poly1d(a)
            with pytest.raises(ValueError):
                b[-1] = 20

    @testing.for_all_dtypes()
    def test_poly1d_get1(self, dtype):
        a1 = testing.shaped_arange((10,), cupy, dtype)
        a2 = testing.shaped_arange((10,), numpy, dtype)
        b1 = cupy.poly1d(a1, variable='z').get()
        b2 = numpy.poly1d(a2, variable='z')
        assert b1 == b2

    @testing.for_all_dtypes()
    def test_poly1d_get2(self, dtype):
        a1 = testing.shaped_arange((), cupy, dtype)
        a2 = testing.shaped_arange((), numpy, dtype)
        b1 = cupy.poly1d(a1).get()
        b2 = numpy.poly1d(a2)
        assert b1 == b2

    @testing.for_all_dtypes(no_bool=True)
    def test_poly1d_set(self, dtype):
        arr1 = testing.shaped_arange((10,), cupy, dtype)
        arr2 = numpy.ones(10, dtype=dtype)
        a = cupy.poly1d(arr1)
        b = numpy.poly1d(arr2, variable='z')
        a.set(b)
        assert a.variable == b.variable
        testing.assert_array_equal(a.coeffs, b.coeffs)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_repr(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return repr(xp.poly1d(a))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_str(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return str(xp.poly1d(a))


@testing.gpu
class TestPoly1dArithmetic(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_mul_scalar(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = xp.poly1d(a) * 10
        return b.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_div_scalar(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        b = xp.poly1d(a) / 10
        return b.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_add_poly1d_array(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = xp.poly1d(a) + a
        return b.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_add_poly1d_poly1d(self, xp, dtype):
        a1 = testing.shaped_arange((5,), xp, dtype)
        b1 = xp.poly1d(a1, variable='z')
        a2 = testing.shaped_arange((2,), xp, dtype)
        b2 = xp.poly1d(a2)
        b1 += b2
        assert b1.variable == 'x'
        return b1.coeffs


@testing.gpu
class TestPoly1dEquality(unittest.TestCase):

    def make_poly1d1(self, xp, dtype):
        a1 = testing.shaped_arange((4,), xp, dtype)
        a2 = xp.zeros((4,), dtype)
        b1 = xp.poly1d(a1)
        b2 = xp.poly1d(a2)
        return b1, b2

    def make_poly1d2(self, xp, dtype):
        a1 = testing.shaped_arange((4,), xp, dtype)
        a2 = testing.shaped_arange((4,), xp, dtype)
        b1 = xp.poly1d(a1)
        b2 = xp.poly1d(a2)
        return b1, b2

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_eq1(self, xp, dtype):
        a, b = self.make_poly1d1(xp, dtype)
        return a == b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_eq2(self, xp, dtype):
        a, b = self.make_poly1d2(xp, dtype)
        return a == b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_ne1(self, xp, dtype):
        a, b = self.make_poly1d1(xp, dtype)
        return a != b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_poly1d_ne2(self, xp, dtype):
        a, b = self.make_poly1d2(xp, dtype)
        return a != b


@testing.gpu
@testing.parameterize(*testing.product({
    'shape1': [(), (0,), (3,), (5,)],
    'shape2': [(), (0,), (3,), (5,)]
}))
class TestPolyaddShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_polyadd(self, xp, dtype):
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return xp.polyadd(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape1': [(3, 3, 1), (3, 3), (3, 1), (3,), (1,)],
    'shape2': [(3, 3, 1), (3, 3), (3, 1), (3,), (1,)]
}))
class TestPolyaddNdimShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_polyadd(self, xp, dtype):
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return xp.polyadd(a, b)


@testing.gpu
class TestPolyadd(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_polyadd_list(self, xp):
        a = [1, 2, 3]
        b = [4, 2]
        return xp.polyadd(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_polyadd_scalar_poly1d(self, xp, dtype):
        a = dtype(10)
        b = testing.shaped_arange((5,), xp, dtype)
        b = xp.poly1d(b)
        return xp.polyadd(a, b).coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_polyadd_poly1d_poly1d(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = xp.poly1d(a, variable='z')
        c = xp.polyadd(b, b)
        assert c.variable == 'x'
        return c.coeffs

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_polyadd_array_poly1d(self, xp, dtype):
        a = testing.shaped_arange((3,), xp, dtype)
        b = testing.shaped_arange((5,), xp, dtype)
        b = xp.poly1d(b)
        return xp.polyadd(a, b).coeffs

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_polyadd_diff_types_array(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((10,), xp, dtype1)
        b = testing.shaped_arange((5,), xp, dtype2)
        return xp.polyadd(a, b)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_polyadd_diff_types_poly1d(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((10,), xp, dtype1)
        b = testing.shaped_arange((5,), xp, dtype2)
        a = xp.poly1d(a, variable='z')
        b = xp.poly1d(b, variable='y')
        c = xp.polyadd(a, b)
        assert c.variable == 'x'
        return c.coeffs


@testing.gpu
class TestPolyaddInvalid(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_polyadd_ndim1(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4, 2), xp, dtype)
            b = testing.shaped_arange((4, 2), xp, dtype)
            with pytest.raises(ValueError):
                xp.polyadd(a, b)

    @testing.for_all_dtypes()
    def test_polyadd_ndim2(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4, 2), xp, dtype)
            b = testing.shaped_arange((3, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.polyadd(a, b)

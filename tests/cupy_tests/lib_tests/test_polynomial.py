import unittest

import pytest
import numpy

import cupy
from cupy.cuda import runtime
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
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_cupy_array(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out

    @testing.numpy_cupy_array_equal()
    def test_poly1d_list(self, xp):
        with cupyx.allow_synchronize(False):
            out = xp.poly1d([1, 2, 3, 4], variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_numpy_poly1d(self, xp, dtype):
        array = testing.shaped_arange((5,), numpy, dtype)
        a = numpy.poly1d(array)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_numpy_poly1d_variable(self, xp, dtype):
        array = testing.shaped_arange((5,), numpy, dtype)
        a = numpy.poly1d(array, variable='z')
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'z')
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_cupy_poly1d(self, xp, dtype):
        array = testing.shaped_arange((5,), xp, dtype)
        a = xp.poly1d(array)
        out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_cupy_poly1d_variable(self, xp, dtype):
        array = testing.shaped_arange((5,), xp, dtype)
        a = xp.poly1d(array, variable='z')
        out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'z')
        return out

    @testing.with_requires('numpy>=1.20')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_zero_dim(self, xp, dtype):
        a = testing.shaped_arange((), xp, dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out

    @testing.with_requires('numpy>=1.20')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_zero_size(self, xp, dtype):
        a = testing.shaped_arange((0,), xp, dtype)
        with cupyx.allow_synchronize(False):
            out = xp.poly1d(a, variable=self.variable)
        assert out.variable == (self.variable or 'x')
        return out


@testing.gpu
class TestPoly1d(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_leading_zeros(self, xp, dtype):
        a = xp.array([0, 0, 1, 2, 3], dtype)
        return xp.poly1d(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_poly1d_neg(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return -xp.poly1d(a)

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

    @pytest.mark.xfail(runtime.is_hip, reason='rocBLAS not implemented')
    @testing.for_signed_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_poly1d_roots(self, xp, dtype):
        a = xp.array([-3, -2.5, 3], dtype)
        out = xp.poly1d(a).roots
        # The current `cupy.roots` doesn't guarantee the order of results.
        return xp.sort(out)

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
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_poly1d_setitem_leading_zeros(self, xp, dtype):
        a = xp.array([0, 0, 0, 2, 3, 0], dtype)
        b = xp.poly1d(a)
        with cupyx.allow_synchronize(False):
            b[1] = 10
        return b

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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_poly1d_call(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = xp.poly1d(a)
        return b(a)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(), (0,), (5,)],
    'exp': [0, 4, 5, numpy.int32(5), numpy.int64(5)],
}))
class TestPoly1dPow(unittest.TestCase):

    @testing.with_requires('numpy>=1.20')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-1)
    def test_poly1d_pow_scalar(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.poly1d(a) ** self.exp


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(5,), (5, 2)],
    'exp': [-10, 3.5, [1, 2, 3]],
}))
class TestPoly1dPowInvalidValue(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_poly1d_pow(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(ValueError):
                xp.poly1d(a) ** self.exp


@testing.gpu
@testing.parameterize(*testing.product({
    'exp': [3.0, numpy.float64(5)],
}))
class TestPoly1dPowInvalidType(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_poly1d_pow(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(TypeError):
                xp.poly1d(a) ** self.exp


class Poly1dTestBase(unittest.TestCase):

    def _get_input(self, xp, in_type, dtype):
        if in_type == 'poly1d':
            return xp.poly1d(testing.shaped_arange((10,), xp, dtype) + 1)
        if in_type == 'ndarray':
            return testing.shaped_arange((10,), xp, dtype)
        if in_type == 'python_scalar':
            return dtype(5).item()
        if in_type == 'numpy_scalar':
            return dtype(5)
        assert False


@testing.gpu
@testing.parameterize(*testing.product({
    'func': [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
    ],
    'type_l': ['poly1d', 'python_scalar', 'ndarray'],
    'type_r': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
}))
class TestPoly1dPolynomialArithmetic(Poly1dTestBase):

    @testing.with_requires('numpy>=1.20')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, accept_error=TypeError)
    def test_poly1d_arithmetic(self, xp, dtype):
        a1 = self._get_input(xp, self.type_l, dtype)
        a2 = self._get_input(xp, self.type_r, dtype)
        return self.func(a1, a2)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['add', 'subtract', 'multiply', 'divide', 'power'],
    'type_l': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
    'type_r': ['poly1d'],
}))
@pytest.mark.xfail(runtime.is_hip,
                   reason='HIP/ROCm does not support cuda array interface')
class TestPoly1dMathArithmetic(Poly1dTestBase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_poly1d_arithmetic(self, xp, dtype):
        func = getattr(xp, self.fname)
        a1 = self._get_input(xp, self.type_l, dtype)
        a2 = self._get_input(xp, self.type_r, dtype)
        return func(a1, a2)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
    ],
    'type_l': ['numpy_scalar'],
    'type_r': ['poly1d'],
}))
class TestPoly1dArithmeticInvalid(Poly1dTestBase):

    @testing.for_all_dtypes()
    def test_poly1d_arithmetic_invalid(self, dtype):
        # CuPy does not support them because device-to-host synchronization is
        # needed to convert the return value to cupy.ndarray type.
        n1 = self._get_input(numpy, self.type_l, dtype)
        n2 = self._get_input(numpy, self.type_r, dtype)
        assert type(self.func(n1, n2)) is numpy.ndarray

        c1 = self._get_input(cupy, self.type_l, dtype)
        c2 = self._get_input(cupy, self.type_r, dtype)
        with pytest.raises(TypeError):
            self.func(c1, c2)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['polyadd', 'polysub', 'polymul'],
    'type_l': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
    'type_r': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
}))
class TestPoly1dRoutines(Poly1dTestBase):

    @testing.with_requires('numpy>=1.20')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, accept_error=TypeError)
    def test_poly1d_routine(self, xp, dtype):
        func = getattr(xp, self.fname)
        a1 = self._get_input(xp, self.type_l, dtype)
        a2 = self._get_input(xp, self.type_r, dtype)
        return func(a1, a2)


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
    'fname': ['polyadd', 'polysub', 'polymul'],
    'shape1': [(), (0,), (3,), (5,)],
    'shape2': [(), (0,), (3,), (5,)],
}))
class TestPolyArithmeticShapeCombination(unittest.TestCase):

    @testing.with_requires('numpy>=1.20')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyroutine(self, xp, dtype):
        func = getattr(xp, self.fname)
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return func(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['polyadd', 'polysub', 'polymul'],
}))
class TestPolyArithmeticDiffTypes(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    def test_polyroutine_diff_types_array(self, dtype1, dtype2):
        def f(xp):
            func = getattr(xp, self.fname)
            a = testing.shaped_arange((10,), xp, dtype1)
            b = testing.shaped_arange((5,), xp, dtype2)
            return func(a, b)
        rtol = 1e-5
        if runtime.is_hip and self.fname == 'polymul':
            rtol = 1e-4
        try:
            testing.assert_allclose(f(cupy), f(numpy), rtol=rtol)
        except TypeError:
            pass

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    def test_polyroutine_diff_types_poly1d(self, dtype1, dtype2):
        def f(xp):
            func = getattr(xp, self.fname)
            a = testing.shaped_arange((10,), xp, dtype1)
            b = testing.shaped_arange((5,), xp, dtype2)
            a = xp.poly1d(a, variable='z')
            b = xp.poly1d(b, variable='y')
            out = func(a, b)
            assert out.variable == 'x'
            return out
        rtol = 1e-5
        if runtime.is_hip and self.fname == 'polymul':
            rtol = 1e-4
        try:
            testing.assert_allclose(f(cupy), f(numpy), rtol=rtol)
        except TypeError:
            pass


@testing.gpu
@testing.parameterize(*testing.product({
    'shape1': [(3,)],
    'shape2': [(3,), (3, 2)],
    'deg': [0, 3, 3.0, 3.5, 10],
    'rcond': [None, 0.5, 1e-15],
    'weighted': [True, False]
}))
class TestPolyfitParametersCombinations(unittest.TestCase):

    def _full_fit(self, xp, dtype):
        x = testing.shaped_arange(self.shape1, xp, dtype)
        y = testing.shaped_arange(self.shape2, xp, dtype)
        w = x if self.weighted else None
        return xp.polyfit(x, y, self.deg, self.rcond, True, w)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        atol=1e-9, accept_error=TypeError, contiguous_check=False)
    def test_polyfit_default(self, xp, dtype):
        if runtime.is_hip and self.deg == 0:
            pytest.xfail('ROCm/HIP may have a bug')
        x = testing.shaped_arange(self.shape1, xp, dtype)
        y = testing.shaped_arange(self.shape2, xp, dtype)
        w = x if self.weighted else None
        return xp.polyfit(x, y, self.deg, self.rcond, w=w)

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_full(self, dtype):
        if runtime.is_hip and self.deg == 0:
            pytest.xfail('ROCm/HIP may have a bug')

        cp_c, cp_resids, cp_rank, cp_s, cp_rcond = self._full_fit(cupy, dtype)
        np_c, np_resids, np_rank, np_s, np_rcond = self._full_fit(numpy, dtype)

        testing.assert_allclose(cp_c, np_c, atol=1e-9)
        testing.assert_allclose(cp_resids, np_resids, atol=1e-9)
        testing.assert_allclose(cp_s, np_s, atol=1e-9)
        assert cp_rank == np_rank
        if self.rcond is not None:
            assert cp_rcond == np_rcond


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(3,), (3, 2)],
    'deg': [0, 1],
    'rcond': [None, 1e-15],
    'weighted': [True, False],
    'cov': ['unscaled', True]
}))
class TestPolyfitCovMode(unittest.TestCase):

    def _cov_fit(self, xp, dtype):
        x = xp.array([0.008, 0.01, 0.015], dtype)
        y = testing.shaped_arange(self.shape, xp, dtype)
        w = x if self.weighted else None
        return xp.polyfit(x, y, self.deg, self.rcond, w=w, cov=self.cov)

    @testing.for_float_dtypes(no_float16=True)
    def test_polyfit_cov(self, dtype):
        if runtime.is_hip and self.deg == 0:
            pytest.xfail('ROCm/HIP may have a bug')
        cp_c, cp_cov = self._cov_fit(cupy, dtype)
        np_c, np_cov = self._cov_fit(numpy, dtype)
        testing.assert_allclose(cp_c, np_c, rtol=1e-5)
        testing.assert_allclose(cp_cov, np_cov, rtol=1e-3)


@testing.gpu
class TestPolyfit(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_poor_fit(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, dtype)
            y = testing.shaped_arange((5,), xp, dtype)
            with pytest.warns(numpy.RankWarning):
                xp.polyfit(x, y, 6)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(), (0,), (5, 3, 3)],
}))
class TestPolyfitInvalidShapes(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_x_invalid_shape(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange(self.shape, xp, dtype)
            y = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polyfit(x, y, 5)

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_y_invalid_shape(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, dtype)
            y = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(TypeError):
                xp.polyfit(x, y, 5)

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_w_invalid_shape(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, dtype)
            w = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(TypeError):
                xp.polyfit(x, x, 5, w=w)


@testing.gpu
class TestPolyfitInvalid(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_neg_degree(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, dtype)
            y = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polyfit(x, y, -4)

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_complex_degree(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, dtype)
            y = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polyfit(x, y, 5j)

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_xy_mismatched_length(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((10,), xp, dtype)
            y = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polyfit(x, y, 5)

    @testing.for_all_dtypes(no_float16=True)
    def test_polyfit_yw_mismatched_length(self, dtype):
        for xp in (numpy, cupy):
            y = testing.shaped_arange((10,), xp, dtype)
            w = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(TypeError):
                xp.polyfit(y, y, 5, w=w)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    def test_polyfit_cov_invalid(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, dtype)
            y = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polyfit(x, y, 5, cov=True)

    def test_polyfit_float16(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5,), xp, numpy.float16)
            with pytest.raises(TypeError):
                xp.polyfit(x, x, 4)


@testing.gpu
class TestPolyfitDiffTypes(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True, full=True)
    @testing.numpy_cupy_allclose(rtol=1e-1, atol=1e-1, accept_error=TypeError)
    def test_polyfit_unweighted_diff_types(self, xp, dtype1, dtype2):
        x = testing.shaped_arange((5,), xp, dtype1)
        y = testing.shaped_arange((5,), xp, dtype2)
        return xp.polyfit(x, y, 5)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2', 'dtype3'], no_bool=True, full=True)
    @testing.numpy_cupy_allclose(atol=1e-0, accept_error=TypeError)
    def test_polyfit_weighted_diff_types(self, xp, dtype1, dtype2, dtype3):
        x = testing.shaped_arange((5,), xp, dtype1)
        y = testing.shaped_arange((5,), xp, dtype2)
        w = testing.shaped_arange((5,), xp, dtype3)
        return xp.polyfit(x, y, 5, w=w)


@testing.gpu
@testing.parameterize(*testing.product({
    'type_l': ['poly1d', 'ndarray'],
    'type_r': ['ndarray', 'numpy_scalar', 'python_scalar'],
}))
class TestPolyval(Poly1dTestBase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_polyval(self, xp, dtype):
        a1 = self._get_input(xp, self.type_l, dtype)
        a2 = self._get_input(xp, self.type_r, dtype)
        return xp.polyval(a1, a2)


@testing.gpu
@testing.parameterize(*testing.product({
    'type_l': ['numpy_scalar', 'python_scalar'],
    'type_r': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
}))
class TestPolyvalInvalidTypes(Poly1dTestBase):

    @testing.for_all_dtypes()
    def test_polyval(self, dtype):
        for xp in (numpy, cupy):
            a1 = self._get_input(xp, self.type_l, dtype)
            a2 = self._get_input(xp, self.type_r, dtype)
            with pytest.raises(TypeError):
                xp.polyval(a1, a2)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape1': [(0,), (3,), (5,)],
    'shape2': [(), (0,), (3,), (5,)]
}))
class TestPolyvalShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyval(self, xp, dtype):
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return xp.polyval(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(), (0,), (3,), (5,)]
}))
class TestPolyvalInvalidShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_polyval(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            b = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(TypeError):
                xp.polyval(a, b)


@testing.gpu
class TestPolyvalDtypesCombination(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'], full=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyval_diff_types_array_array(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((10,), xp, dtype1)
        b = testing.shaped_arange((5,), xp, dtype2)
        return xp.polyval(a, b)

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'], full=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_polyval_diff_types_array_scalar(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((10,), xp, dtype1)
        b = dtype2(5)
        return xp.polyval(a, b)


@testing.gpu
class TestPolyvalNotImplemented(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_polyval_ndim_values(self, dtype):
        a = testing.shaped_arange((2, ), cupy, dtype)
        b = testing.shaped_arange((2, 4), cupy, dtype)
        with pytest.raises(NotImplementedError):
            cupy.polyval(a, b)

    @testing.for_all_dtypes()
    def test_polyval_poly1d_values(self, dtype):
        a = testing.shaped_arange((5,), cupy, dtype)
        b = testing.shaped_arange((3,), cupy, dtype)
        b = cupy.poly1d(b)
        with pytest.raises(NotImplementedError):
            cupy.polyval(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['polyadd', 'polysub', 'polymul', 'polyval'],
}))
class TestPolyRoutinesNdim(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_polyroutine_ndim(self, dtype):
        for xp in (numpy, cupy):
            func = getattr(xp, self.fname)
            a = testing.shaped_arange((2, 3, 4), xp, dtype)
            b = testing.shaped_arange((10, 5), xp, dtype)
            with pytest.raises(ValueError):
                func(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'input': [[2, -1, -2], [-4, 10, 4]],
}))
@pytest.mark.xfail(runtime.is_hip, reason='rocBLAS not implemented')
class TestRootsReal(unittest.TestCase):

    @testing.for_signed_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_roots_array(self, xp, dtype):
        a = xp.array(self.input, dtype)
        out = xp.roots(a)
        return xp.sort(out)

    @testing.for_signed_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_roots_poly1d(self, xp, dtype):
        a = xp.array(self.input, dtype)
        out = xp.roots(xp.poly1d(a))
        return xp.sort(out)


@testing.gpu
@testing.parameterize(*testing.product({
    'input': [[3j, 1.5j, -3j], [3 + 2j, 5], [3j, 0], [0, 3j]],
}))
class TestRootsComplex(unittest.TestCase):

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_roots_array(self, xp, dtype):
        if runtime.is_hip and self.input == [3j, 1.5j, -3j]:
            pytest.xfail('rocBLAS not implemented')
        a = xp.array(self.input, dtype)
        out = xp.roots(a)
        return xp.sort(out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_roots_poly1d(self, xp, dtype):
        if runtime.is_hip and self.input == [3j, 1.5j, -3j]:
            pytest.xfail('rocBLAS not implemented')
        a = xp.array(self.input, dtype)
        out = xp.roots(xp.poly1d(a))
        return xp.sort(out)


@testing.gpu
@testing.parameterize(*testing.product({
    'input': [[5, 10], [5, 0], [0, 5], [0, 0], [5]],
}))
class TestRootsSpecialCases(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_roots_array(self, xp, dtype):
        a = xp.array(self.input, dtype)
        return xp.roots(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_roots_poly1d(self, xp, dtype):
        a = xp.array(self.input, dtype)
        return xp.roots(xp.poly1d(a))


@testing.gpu
class TestRoots(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_roots_zero_sized(self, xp, dtype):
        a = xp.zeros((0,), dtype)
        return xp.roots(a)

    @testing.with_requires('numpy>1.17')
    @testing.for_all_dtypes(no_bool=True)
    def test_roots_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp, dtype)
            with pytest.raises(TypeError):
                xp.roots(a)

    @testing.for_all_dtypes(no_bool=True)
    def test_roots_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 1), xp, dtype)
            with pytest.raises(ValueError):
                xp.roots(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_roots_zeros(self, xp, dtype):
        a = xp.zeros((3,), dtype)
        return xp.roots(a)

    @testing.for_all_dtypes(no_bool=True)
    def test_roots_zeros_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros((2, 1), dtype)
            with pytest.raises(ValueError):
                xp.roots(a)

    def test_roots_bool_symmetric(self):
        a = cupy.array([5, -1, -5], bool)
        with pytest.raises(NotImplementedError):
            cupy.roots(a)

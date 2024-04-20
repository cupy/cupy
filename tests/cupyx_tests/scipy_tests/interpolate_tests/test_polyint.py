
import io
import warnings

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing
import cupyx.scipy.interpolate  # NOQA
from cupyx.scipy.interpolate import CubicHermiteSpline

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


if cupy.cuda.runtime.runtimeGetVersion() < 11000:
    # Workarounds precison issues in CUDA 10.2 + float16
    default_atol = 5e-2
    default_rtol = 1e-1
else:
    default_atol = 0
    default_rtol = 1e-7


@testing.with_requires("scipy")
class TestBarycentric:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_lagrange(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 100, dtype=dtype)
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-4)
    def test_scalar(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'iu':
            pytest.skip()
        true_poly = numpy.poly1d([-1, 2, 6, -3, 2])
        xs = numpy.linspace(-1, 1, 10, dtype=dtype)
        ys = true_poly(xs)
        if xp is cupy:
            xs = cupy.asarray(xs)
            ys = cupy.asarray(ys)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(xp.array(7, dtype=dtype))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_delayed(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 100, dtype=dtype)
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs)
        P.set_yi(ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_append(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 100, dtype=dtype)
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs[:3], ys[:3])
        P.add_xi(xs[3:], ys[3:])
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vector(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        xs = xp.array([0, 1, 2], dtype=dtype)
        ys = xp.array([[0, 1], [1, 0], [2, 1]], dtype=dtype)
        test_xs = xp.linspace(-1, 3, 100, dtype=dtype)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(test_xs)

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_scalarvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-10, 10, 20)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
            xs, xp.outer(ys, xp.arange(3)))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_1d_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
            xs, xp.outer(ys, xp.array([1])))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @testing.with_requires("scipy>=1.8.0")
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_large_chebyshev(self, xp, scp, dtype):
        n = 100
        j = numpy.arange(n + 1, dtype=dtype).astype(numpy.float64)
        x = numpy.cos(j * numpy.pi / n)

        if xp is cupy:
            j = cupy.asarray(j)
            x = cupy.asarray(x)
        # The weights for Chebyshev points against SciPy counterpart
        return scp.interpolate.BarycentricInterpolator(x).wi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex(self, xp, scp):
        x = xp.array([1, 2, 3, 4])
        y = xp.array([1, 2, 1j, 3])
        xi = xp.array([0, 8, 1, 5])
        return scp.interpolate.BarycentricInterpolator(x, y)(xi)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_wrapper(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-2, 2, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.barycentric_interpolate(xs, ys, test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_array_input(self, xp, scp, dtype):
        x = 1000 * xp.arange(1, 11, dtype=dtype)
        y = xp.arange(1, 11, dtype=dtype)
        xi = xp.array(1000 * 9.5)
        return scp.interpolate.barycentric_interpolate(x, y, xi)


@testing.with_requires("scipy")
class TestKrogh:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_lagrange(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        out = P(test_xs)
        print(out.dtype)
        return out

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-6, rtol=1e-6)
    def test_scalar(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = numpy.poly1d([-1, 2, 6, -3, 2])
        xs = numpy.linspace(-1, 1, 10, dtype=dtype)
        ys = true_poly(xs)
        if xp is cupy:
            xs = cupy.asarray(xs)
            ys = cupy.asarray(ys)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(xp.array(7, dtype=dtype))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_derivatives(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P.derivatives(test_xs)
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_low_derivatives(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P.derivatives(test_xs, len(xs) + 2)
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        m = 10
        return [P.derivative(test_xs, i) for i in range(m)]

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_high_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P.derivative(test_xs, 2 * len(xs))
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_hermite(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        D = P(test_xs)
        return D

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_hermite_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        xs = xp.array([0, 0, 0], dtype=dtype)
        ys = xp.array([1, 2, 3], dtype=dtype)
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_vector(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        xs = xp.array([0, 1, 2], dtype=dtype)
        ys = xp.array([[0, 1], [1, 0], [2, 1]], dtype=dtype)
        test_xs = xp.linspace(-1, 3, 5, dtype=dtype)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(test_xs)

    @testing.for_dtypes('bhilfd')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty(self, xp, scp, dtype):
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        xs = xp.linspace(-5, 5, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(xp.array([]))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_scalarvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_scalarvalue_derivatives(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        test_xs = xp.array(test_xs)
        return xp.shape(P.derivatives(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-10, 10, 20)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(
            xs, xp.outer(ys, xp.arange(3)))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_vectorvalue_derivative(self, xp, scp, test_xs):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-10, 10, 20)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(
            xs, xp.outer(ys, xp.arange(3)))
        test_xs = xp.array(test_xs)
        return xp.shape(P.derivatives(test_xs))

    @pytest.mark.parametrize('test_xs', [0, [0], [0, 1]])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_shapes_1d_vectorvalue(self, xp, scp, test_xs):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8])
        xs = xp.linspace(-1, 10, 10)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(
            xs, xp.outer(ys, xp.array([1])))
        test_xs = xp.array(test_xs)
        return xp.shape(P(test_xs))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_wrapper(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.krogh_interpolate(xs, ys, test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=default_atol, rtol=default_rtol, scipy_name='scp')
    def test_wrapper2(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.krogh_interpolate(xs, ys, test_xs, der=3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivatives_complex(self, xp, scp):
        x = xp.array([-1, -1, 0, 1, 1])
        y = xp.array([1, 1.0j, 0, -1, 1.0j])
        P = scp.interpolate.KroghInterpolator(x, y)
        D = P.derivatives(xp.array(0))
        return D


@testing.with_requires("scipy>=1.10.0")
class TestZeroSizeArrays:
    # regression tests for gh-17241 : CubicSpline et al must not segfault
    # when y.size == 0
    # The two methods below are _almost_ the same, but not quite:
    # one is for objects which have the `bc_type` argument (CubicSpline)
    # and the other one is for those which do not (Pchip, Akima1D)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('y_shape', [(10, 0, 5), (10, 5, 0)])
    @pytest.mark.parametrize('bc_type',
                             ['not-a-knot', 'periodic', 'natural', 'clamped'])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('klass', ['make_interp_spline', 'CubicSpline'])
    def test_zero_size(self, xp, scp, klass, y_shape, bc_type, axis):
        if runtime.is_hip and bc_type == 'periodic':
            pytest.xfail('Not implemented on HIP/ROCm')
        x = xp.arange(10)
        y = xp.zeros(y_shape)
        xval = xp.arange(3)

        cls = getattr(scp.interpolate, klass)
        obj = cls(x, y, bc_type=bc_type)
        r1 = obj(xval)
        assert r1.size == 0
        assert r1.shape == xval.shape + y.shape[1:]

        # Also check with an explicit non-default axis
        yt = xp.moveaxis(y, 0, axis)  # (10, 0, 5) --> (0, 10, 5) if axis=1 etc

        obj = cls(x, yt, bc_type=bc_type, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        r2 = obj(xval)
        assert r2.size == 0
        assert r2.shape == sh
        return r1, r2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('y_shape', [(10, 0, 5), (10, 5, 0)])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('klass',
                             ['PchipInterpolator', 'Akima1DInterpolator'])
    def test_zero_size_2(self, xp, scp, klass, y_shape, axis):
        x = xp.arange(10)
        y = xp.zeros(y_shape)
        xval = xp.arange(3)

        cls = getattr(scp.interpolate, klass)
        obj = cls(x, y)
        r1 = obj(xval)
        assert r1.size == 0
        assert r1.shape == xval.shape + y.shape[1:]

        # Also check with an explicit non-default axis
        yt = xp.moveaxis(y, 0, axis)  # (10, 0, 5) --> (0, 10, 5) if axis=1 etc

        obj = cls(x, yt, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        r2 = obj(xval)
        assert r2.size == 0
        assert r2.shape == sh
        return r1, r2


@testing.with_requires("scipy")
class TestCubicHermiteSpline:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_correctness(self, xp, scp):
        x = xp.asarray([0, 2, 7])
        y = xp.asarray([-1, 2, 3])
        dydx = xp.asarray([0, 3, 7])
        s = scp.interpolate.CubicHermiteSpline(x, y, dydx)
        return s(x), s(x, 1)

    def test_ctor_error_handling(self):
        x = cupy.asarray([1, 2, 3])
        y = cupy.asarray([0, 3, 5])
        dydx = cupy.asarray([1, -1, 2, 3])
        dydx_with_nan = cupy.asarray([1, 0, cupy.nan])

        with pytest.raises(ValueError):
            CubicHermiteSpline(x, y, dydx)

        with pytest.raises(ValueError):
            CubicHermiteSpline(x, y, dydx_with_nan)


@testing.with_requires("scipy")
class TestPCHIP:
    def _make_random(self, xp, scp, npts=20):
        xi = xp.sort(testing.shaped_random((npts,), xp))
        yi = testing.shaped_random((npts,), xp)
        return scp.interpolate.PchipInterpolator(xi, yi), xi, yi

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-3)
    def test_overshoot(self, xp, scp):
        # PCHIP should not overshoot
        p, xi, _ = self._make_random(xp, scp)
        results = []
        for i in range(len(xi) - 1):
            x1, x2 = xi[i], xi[i+1]
            x = xp.linspace(x1, x2, 10)
            yp = p(x)
            results.append(yp)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-3)
    def test_monotone(self, xp, scp):
        # PCHIP should preserve monotonicty
        p, xi, _ = self._make_random(xp, scp)
        results = []
        for i in range(len(xi) - 1):
            x1, x2 = xi[i], xi[i+1]
            x = xp.linspace(x1, x2, 10)
            yp = p(x)
            results.append(yp)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cast(self, xp, scp):
        # regression test for integer input data, see gh-3453
        data = xp.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100],
                         [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
        xx = xp.arange(100)
        curve = scp.interpolate.PchipInterpolator(data[0], data[1])(xx)

        data1 = data * 1.0
        curve1 = scp.interpolate.PchipInterpolator(data1[0], data1[1])(xx)
        return curve, curve1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nag(self, xp, scp):
        # Example from NAG C implementation,
        # http://nag.com/numeric/cl/nagdoc_cl25/html/e01/e01bec.html
        # suggested in scipy/gh-5326 as a smoke test for the way the
        # derivatives are computed (see also scipy/gh-3453)
        dataStr = '''
          7.99   0.00000E+0
          8.09   0.27643E-4
          8.19   0.43750E-1
          8.70   0.16918E+0
          9.20   0.46943E+0
         10.00   0.94374E+0
         12.00   0.99864E+0
         15.00   0.99992E+0
         20.00   0.99999E+0
        '''
        data = xp.loadtxt(io.StringIO(dataStr))
        pch = scp.interpolate.PchipInterpolator(data[:, 0], data[:, 1])

        resultStr = '''
           7.9900       0.0000
           9.1910       0.4640
          10.3920       0.9645
          11.5930       0.9965
          12.7940       0.9992
          13.9950       0.9998
          15.1960       0.9999
          16.3970       1.0000
          17.5980       1.0000
          18.7990       1.0000
          20.0000       1.0000
        '''
        result = xp.loadtxt(io.StringIO(resultStr))
        return pch(result[:, 0])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_endslopes(self, xp, scp):
        # this is a smoke test for scipy/gh-3453: PCHIP interpolator should not
        # set edge slopes to zero if the data do not suggest zero
        # edge derivatives
        x = xp.array([0.0, 0.1, 0.25, 0.35])
        y1 = xp.array([279.35, 0.5e3, 1.0e3, 2.5e3])
        y2 = xp.array([279.35, 2.5e3, 1.50e3, 1.0e3])
        pchip = scp.interpolate.PchipInterpolator
        results = []
        for pp in (pchip(x, y1), pchip(x, y2)):
            for t in (x[0], x[-1]):
                results.append(pp(t, 1))
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_all_zeros(self, xp, scp):
        x = xp.arange(10)
        y = xp.zeros_like(x)

        # this should work and not generate any warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            pch = scp.interpolate.PchipInterpolator(x, y)

        xx = xp.linspace(0, 9, 101)
        return pch(xx)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_two_points(self, xp, scp):
        # regression test for gh-6222: pchip([0, 1], [0, 1]) fails because
        # it tries to use a three-point scheme to estimate edge derivatives,
        # while there are only two points available.
        # Instead, it should construct a linear interpolator.
        x = xp.linspace(0, 1, 11)
        p = scp.interpolate.PchipInterpolator([0, 1], [0, 2])
        return p(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pchip_interpolate(self, xp, scp):
        r1 = scp.interpolate.pchip_interpolate(
            [1, 2, 3], [4, 5, 6], [0.5], der=1)
        r2 = scp.interpolate.pchip_interpolate(
            [1, 2, 3], [4, 5, 6], [0.5], der=0)
        r3 = scp.interpolate.pchip_interpolate(
            [1, 2, 3], [4, 5, 6], [0.5], der=[0, 1])
        return r1, r2, xp.asarray(r3)


@testing.with_requires("scipy")
class TestAkima1D:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_correctness(self, xp, scp):
        x = xp.asarray([-1, 0, 1, 2, 3, 4])
        # y = xp.asarray([-1, 2, 3])
        y = testing.shaped_random((6, 1), xp)
        s = scp.interpolate.Akima1DInterpolator(x, y)
        return s(x), s(x, 1)

@pytest.mark.skipif(runtime.is_hip, reason='Currently unsupported on ROCm/HIP')
@testing.with_requires("scipy")
class TestCubicSpline:
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-14)
    @pytest.mark.parametrize('n', [2, 3, 8])    # 8 is x.size
    @pytest.mark.parametrize('bc_type',
                             ['not-a-knot', 'clamped', 'natural',
                                 ((1, 0), (1, 0)), ((1, 0), (2, 1.0))]
                             )
    def test_general(self, xp, scp, n, bc_type):
        x = xp.array([-1, 0, 0.5, 2, 4, 4.5, 5.5, 9])
        y = xp.array([0, -0.5, 2, 3, 2.5, 1, 1, 0.5])

        spl = scp.interpolate.CubicSpline(x[:n], y[:n])
        q = xp.linspace(0, x[:n], 11)
        return spl(q)

    @pytest.mark.parametrize('n', [2, 3, 5])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-14)
    def test_periodic(self, xp, scp, n):
        x = xp.linspace(0, 2 * xp.pi, n)
        y = xp.cos(x)
        S = scp.interpolate.CubicSpline(x, y, bc_type='periodic')
        q = xp.linspace(0, 2*xp.pi, 3*n)
        return S(q)

    @pytest.mark.parametrize('n', [3, 5])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-14)
    def test_periodic_2(self, xp, scp, n):
        x = xp.linspace(0, 2 * xp.pi, n)
        y = xp.cos(x)
        Y = xp.empty((2, n, 2))
        Y[0, :, 0] = y
        Y[0, :, 1] = y + 2
        Y[1, :, 0] = y - 1
        Y[1, :, 1] = y + 5
        S = scp.interpolate.CubicSpline(x, Y, axis=1, bc_type='periodic')
        q = xp.linspace(0, 2*xp.pi, 3*n)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_periodic_eval(self, xp, scp):
        x = xp.linspace(0, 2 * xp.pi, 10)
        y = xp.cos(x)
        S = scp.interpolate.CubicSpline(x, y, bc_type='periodic')
        return S(1), S(1 + 2 * xp.pi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_second_derivative_continuity_gh_11758(self, xp, scp):
        # gh-11758: C2 continuity fail
        x = xp.array([0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0,
                      7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3])
        y = xp.array([1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1,
                      2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4,
                      1.3])
        S = scp.interpolate.CubicSpline(
            x, y, bc_type='periodic', extrapolate='periodic')
        q = xp.linspace(0.9, 13.3, 21)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_three_points(self, xp, scp):
        # gh-11758: Fails computing a_m2_m1
        # In this case, s (first derivatives) could be found manually by
        # solving system of 2 linear equations. Due to solution of this system,
        # s[i] = (h1m2 + h2m1) / (h1 + h2), where
        # h1 = x[1] - x[0], h2 = x[2] - x[1],
        # m1 = (y[1] - y[0]) / h1, m2 = (y[2] - y[1]) / h2
        x = xp.array([1.0, 2.75, 3.0])
        y = xp.array([1.0, 15.0, 1.0])
        S = scp.interpolate.CubicSpline(x, y, bc_type='periodic')
        return S.derivative(1)(x)

    @testing.with_requires("scipy >= 1.13")
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_periodic_three_points_multidim(self, xp, scp):
        # make sure one multidimensional interpolator does the same as multiple
        # one-dimensional interpolators
        x = xp.array([0.0, 1.0, 3.0])
        y = xp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        S = scp.interpolate.CubicSpline(x, y, bc_type="periodic")
        q = xp.linspace(0, 2, 5)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dtypes_1(self, xp, scp):
        x = xp.array([0, 1, 2, 3], dtype=int)
        y = xp.array([-5, 2, 3, 1], dtype=int)
        S = scp.interpolate.CubicSpline(x, y)
        q = xp.linspace(0, 3, 7)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_dtypes_2(self, xp, scp):
        x = xp.array([0, 1, 2, 3], dtype=int)
        y = xp.array([-1+1j, 0.0, 1-1j, 0.5-1.5j])
        S = scp.interpolate.CubicSpline(x, y)
        q = xp.linspace(0, 3, 7)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-14)
    def test_dtypes_3(self, xp, scp):
        x = xp.array([0, 1, 2, 3], dtype=int)
        S = scp.interpolate.CubicSpline(
            x, x ** 3, bc_type=("natural", (1, 2j)))
        q = xp.linspace(0, 3, 7)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dtypes_4(self, xp, scp):
        x = xp.array([0, 1, 2, 3], dtype=int)
        y = xp.array([-5, 2, 3, 1])
        S = scp.interpolate.CubicSpline(
            x, y, bc_type=[(1, 2 + 0.5j), (2, 0.5 - 1j)])
        q = xp.linspace(0, 3, 11)
        return S(q)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_small_dx(self, xp, scp):
        # make sure the sample is the same: generate random variates on CPU
        import numpy as np
        rng = np.random.RandomState(0)
        x = np.sort(rng.uniform(size=100))
        y = 1e4 + rng.uniform(size=100)

        x = xp.asarray(x)
        y = xp.asarray(y)
        S = scp.interpolate.CubicSpline(x, y)
        q = xp.linspace(0, 1, 201)
        return S(q)

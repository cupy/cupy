import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.interpolate  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@testing.with_requires("scipy")
class TestKrogh:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_lagrange(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        return P(test_xs)

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
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
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
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
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
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        m = 10
        for i in range(m):
            return P.derivative(test_xs, i)

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
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
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

    @pytest.mark.parametrize('n', [0, [0], [0, 1]])
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_hermite_derivative1(self, xp, scp, dtype, n):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        x_D = P.derivative(xp.array(n), 0)
        return P(x_D)

    @pytest.mark.parametrize('n', [0, [0], [0, 1]])
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7, rtol=1e-7)
    def test_hermite_derivative2(self, xp, scp, dtype, n):
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-5, 5, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        P = scp.interpolate.KroghInterpolator(xs, ys)
        x_D = P.derivatives(xp.array(n), 1)
        return P(x_D)

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
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_wrapper(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-2, 2, 5, dtype=dtype)
        xs = xp.linspace(-1, 1, 5, dtype=dtype)
        ys = true_poly(xs)
        return scp.interpolate.krogh_interpolate(xs, ys, test_xs)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
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

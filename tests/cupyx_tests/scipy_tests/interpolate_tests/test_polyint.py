import cupy

from cupy import testing
from cupyx.scipy.interpolate import BarycentricInterpolator

from scipy import interpolate  # NOQA


@testing.with_requires("scipy")
class TestBarycentric:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lagrange(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-1, 1, 100)
        xs = xp.linspace(-1, 1, 5)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(test_xs)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_scalar(self, xp, scp):
        true_poly = xp.poly1d([-1, 2, 6, -3, 2])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(xp.array(7))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_scalar_2(self, xp, scp):
        true_poly = xp.poly1d([-1, 2, 6, -3, 2])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(7)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_delayed(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-1, 1, 100)
        xs = xp.linspace(-1, 1, 5)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs)
        P.set_yi(ys)
        return P(test_xs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_append(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-1, 1, 100)
        xs = xp.linspace(-1, 1, 5)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs[:3], ys[:3])
        P.add_xi(xs[3:], ys[3:])
        return P(test_xs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vector(self, xp, scp):
        xs = xp.array([0, 1, 2])
        ys = xp.array([[0, 1], [1, 0], [2, 1]])
        test_xs = xp.linspace(-1, 3, 100)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return P(test_xs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_scalarvalue_1(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3, 5])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return xp.shape(P(0))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_scalarvalue_2(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3, 5])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return xp.shape(P(xp.array(0)))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_scalarvalue_3(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3, 5])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return xp.shape(P([0]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_scalarvalue_4(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 5, 1, -3, 5])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(xs, ys)
        return xp.shape(P([0, 1]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_vectorvalue_1(self, xp, scp):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-1, 1, 20)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
                xs,
                xp.outer(ys, xp.arange(3)))
        return P(0)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_vectorvalue_2(self, xp, scp):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-1, 1, 20)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
                xs,
                xp.outer(ys, xp.arange(3)))
        return P([0])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_vectorvalue_3(self, xp, scp):
        true_poly = xp.poly1d([4, -5, 3, 2, -4])
        xs = xp.linspace(-1, 1, 20)
        ys = true_poly(xs)
        P = scp.interpolate.BarycentricInterpolator(
                xs,
                xp.outer(ys, xp.arange(3)))
        return P([0, 1])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_1d_vectorvalue_1(self, xp, scp):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8, 4])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        return xp.shape(scp.interpolate.BarycentricInterpolator(
                        xs,
                        xp.outer(ys, xp.array([1])))(0))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_1d_vectorvalue_2(self, xp, scp):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8, 4])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        return xp.shape(scp.interpolate.BarycentricInterpolator(
                        xs,
                        xp.outer(ys, xp.array([1])))([0]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes_1d_vectorvalue_3(self, xp, scp):
        true_poly = xp.poly1d([-3, -1, 4, 9, 8, 4])
        xs = xp.linspace(-1, 1, 10)
        ys = true_poly(xs)
        return xp.shape(scp.interpolate.BarycentricInterpolator(
                        xs,
                        xp.outer(ys, xp.array([1])))([0, 1]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_large_chebyshev(self, xp, scp):
        n = 800
        j = cupy.arange(n + 1).astype(cupy.float64)
        x = cupy.cos(j * cupy.pi / n)

        w = (-1) ** j
        w[0] *= 0.5
        w[-1] *= 0.5

        return scp.interpolate.BarycentricInterpolator(x).wi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_1(self, xp, scp):
        x = xp.array([1, 2, 3, 4])
        y = xp.array([1, 2, 1j, 3])
        return scp.interpolate.BarycentricInterpolator(x, y)(x)

    def test_complex_2(self):
        x = cupy.array([1, 2, 3, 4])
        y = cupy.array([1, 2, 1j, 3])
        P = BarycentricInterpolator(x, y)
        testing.assert_allclose(y, P(x))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_wrapper(self, xp, scp):
        true_poly = xp.poly1d([-2, 3, 1, 5, -4])
        test_xs = xp.linspace(-1, 1, 100)
        xs = xp.linspace(-1, 1, 5)
        ys = true_poly(xs)
        return scp.interpolate.barycentric_interpolate(xs, ys, test_xs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_int_input(self, xp, scp):
        x = 1000 * xp.arange(1, 11)
        y = xp.arange(1, 11)
        return scp.interpolate.barycentric_interpolate(x, y, 1000 * 9.5)

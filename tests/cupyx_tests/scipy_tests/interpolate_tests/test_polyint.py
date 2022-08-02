import cupy

from cupy import testing
from cupyx.scipy.interpolate import BarycentricInterpolator


@testing.with_requires("scipy")
class TestBarycentric:

    def setup_method(self):
        self.true_poly = cupy.poly1d([-2, 3, 1, 5, -4])
        self.test_xs = cupy.linspace(-1, 1, 100)
        self.xs = cupy.linspace(-1, 1, 5)
        self.ys = self.true_poly(self.xs)

    def test_lagrange(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        testing.assert_allclose(self.true_poly(self.test_xs),
                                P(self.test_xs))

    def test_scalar(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        testing.assert_allclose(self.true_poly(cupy.array(7)),
                                P(cupy.array(7)))
        testing.assert_allclose(self.true_poly(7), P(7))

    def test_delayed(self):
        P = BarycentricInterpolator(self.xs)
        P.set_yi(self.ys)
        testing.assert_allclose(self.true_poly(self.test_xs),
                                P(self.test_xs))

    def test_append(self):
        P = BarycentricInterpolator(self.xs[:3], self.ys[:3])
        P.add_xi(self.xs[3:], self.ys[3:])
        testing.assert_allclose(self.true_poly(self.test_xs),
                                P(self.test_xs))

    def test_vector(self):
        xs = cupy.array([0, 1, 2])
        ys = cupy.array([[0, 1], [1, 0], [2, 1]])
        BI = BarycentricInterpolator
        P = BI(xs, ys)
        Pi = [BI(xs, ys[:, i]) for i in range(ys.shape[1])]
        test_xs = cupy.linspace(-1, 3, 100)
        testing.assert_allclose(P(test_xs),
                                cupy.asarray([p(test_xs) for p in Pi]).T)

    def test_shapes_scalarvalue(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        testing.assert_allclose(cupy.shape(P(0)), ())
        testing.assert_allclose(cupy.shape(P(cupy.array(0))), ())
        testing.assert_allclose(cupy.shape(P([0])), (1,))
        testing.assert_allclose(cupy.shape(P([0, 1])), (2,))

    def test_shapes_vectorvalue(self):
        P = BarycentricInterpolator(self.xs,
                                    cupy.outer(self.ys, cupy.arange(3)))
        testing.assert_allclose(cupy.shape(P(0)), (3,))
        testing.assert_allclose(cupy.shape(P([0])), (1, 3))
        testing.assert_allclose(cupy.shape(P([0, 1])), (2, 3))

    def test_shapes_1d_vectorvalue(self):
        P = BarycentricInterpolator(self.xs,
                                    cupy.outer(self.ys, cupy.array([1])))
        testing.assert_allclose(cupy.shape(P(0)), (1,))
        testing.assert_allclose(cupy.shape(P([0])), (1, 1))
        testing.assert_allclose(cupy.shape(P([0, 1])), (2, 1))

    def test_large_chebyshev(self):
        n = 800
        j = cupy.arange(n + 1).astype(cupy.float64)
        x = cupy.cos(j * cupy.pi / n)

        w = (-1) ** j
        w[0] *= 0.5
        w[-1] *= 0.5

        P = BarycentricInterpolator(x)
        factor = P.wi[0]
        testing.assert_allclose(P.wi / (2 * factor), w)

    def test_complex(self):
        x = cupy.array([1, 2, 3, 4])
        y = cupy.array([1, 2, 1j, 3])

        P = BarycentricInterpolator(x, y)
        testing.assert_allclose(y, P(x))

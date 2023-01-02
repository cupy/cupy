import cupy
from cupy import poly1d
from cupyx.scipy.interpolate import lagrange


class TestLagrange:
    def test_lagrange(self):
        p = poly1d([5, 2, 1, 4, 3])
        xs = cupy.arange(len(p.coeffs))
        ys = p(xs)
        pl = lagrange(xs, ys)
        assert cupy.allclose(p.coeffs, pl.coeffs)

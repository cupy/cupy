import sys
import unittest
import pytest

import cupy
from cupy import testing

import cupyx.scipy.signal  # NOQA

import numpy as np

try:
    import scipy  # NOQA
except ImportError:
    pass

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'input': [(256, 256), (4, 512), (512, 3)],
    'hrow': [1, 3],
    'hcol': [1, 3],
}))
@testing.with_requires('scipy')
class TestSepFIR2d(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_sepfir2d(self, xp, scp, dtype):
        if sys.platform.startswith('win32') and xp.dtype(dtype).kind in 'c':
            # it's likely a SciPy bug (ValueError: incorrect type), so we
            # do this to effectively skip testing it
            return xp.zeros(10)

        input = testing.shaped_random(self.input, xp, dtype)
        hrow = testing.shaped_random((self.hrow,), xp, dtype)
        hcol = testing.shaped_random((self.hcol,), xp, dtype)
        return scp.signal.sepfir2d(input, hrow, hcol)


@testing.with_requires('scipy')
class TestCSpline:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cspline_zero(self, xp, scp):
        return scp.signal.cspline1d(xp.asarray([0]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cspline_lambda_nonzero(self, xp, scp):
        return scp.signal.cspline1d(xp.asarray([1., 2, 3, 4, 5]), 1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cspline(self, xp, scp):
        return scp.signal.cspline1d(xp.asarray([1., 2, 3, 4, 5]))


@testing.with_requires('scipy')
class TestQSpline:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_qspline_zero(self, xp, scp):
        return scp.signal.qspline1d(xp.asarray([0]))

    def test_qspline_lambda_nonzero(self):
        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            with pytest.raises(ValueError):
                scp.signal.qspline1d(xp.asarray([1., 2, 3, 4, 5]), 1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_qspline(self, xp, scp):
        return scp.signal.qspline1d(xp.asarray([1., 2, 3, 4, 5]))


@testing.with_requires('scipy')
class TestCSplineEval:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cspline_eval_zero(self, xp, scp):
        return scp.signal.cspline1d_eval(xp.asarray([0., 0]), xp.asarray([0.]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cspline_eval(self, xp, scp):
        x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        dx = x[1] - x[0]
        newx = [-6., -5.5, -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1.,
                -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.,
                6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.,
                12.5]
        y = xp.asarray([
            4.216, 6.864, 3.514, 6.203, 6.759, 7.433, 7.874, 5.879,
            1.396, 4.094])
        cj = scp.signal.cspline1d(y)
        return scp.signal.cspline1d_eval(cj, newx, dx=dx, x0=x[0])


@testing.with_requires('scipy')
class TestQSplineEval:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_qspline_eval_zero(self, xp, scp):
        return scp.signal.qspline1d_eval(xp.asarray([0., 0]), xp.asarray([0.]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_qspline_eval(self, xp, scp):
        x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        dx = x[1] - x[0]
        newx = [-6., -5.5, -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1.,
                -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.,
                6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.,
                12.5]
        y = xp.asarray([
            4.216, 6.864, 3.514, 6.203, 6.759, 7.433, 7.874, 5.879,
            1.396, 4.094])
        cj = scp.signal.qspline1d(y)
        return scp.signal.qspline1d_eval(cj, newx, dx=dx, x0=x[0])


@testing.with_requires('scipy')
class TestCSpline2D:
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-4, rtol=1e-4)
    def test_cspline2d_iir1(self, xp, scp):
        image = testing.shaped_random((71, 73), xp, xp.float64,
                                      scale=1, seed=181819142)
        return scp.signal.cspline2d(image, -1.0)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-4, rtol=1e-4)
    def test_cspline2d_iir2(self, xp, scp):
        image = testing.shaped_random((71, 73), xp, xp.float64,
                                      scale=1, seed=181819142)
        return scp.signal.cspline2d(image, 8.0)


@testing.with_requires('scipy')
class TestQSpline2D:
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-4, rtol=1e-4)
    def test_qspline2d_iir1(self, xp, scp):
        image = testing.shaped_random((71, 73), xp, xp.float64,
                                      scale=1, seed=181819142)
        return scp.signal.qspline2d(image)


@testing.with_requires('scipy')
class TestSplineFilter:
    def test_spline_filter_lambda_zero(self):
        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            with pytest.raises(ValueError):
                scp.signal.spline_filter(xp.asarray([0.]), 0)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-4, rtol=1e-4)
    def test_spline_filter(self, xp, scp):
        data = testing.shaped_random(
            (12, 12), xp, xp.float64, scale=1, seed=12457)
        data = 10 * (1 - 2 * data)
        return scp.signal.spline_filter(data, 0)


@testing.with_requires('scipy')
class TestGaussSpline:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_gauss_spline(self, xp, scp):
        return scp.signal.gauss_spline(0.0, 0)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_gauss_spline_list(self, xp, scp):
        knots = [-1.0, 0.0, -1.0]
        return scp.signal.gauss_spline(knots, 3)

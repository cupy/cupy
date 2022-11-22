import cupy
from cupy import testing
from cupy.testing import assert_allclose
import numpy as _np
import cupyx.scipy.interpolate as csi  # NOQA
make_interp_spline = csi._bspline2.make_interp_spline   # XXX
BSpline = csi._bspline.BSpline

import pytest

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass



class TestInterp:
    #
    # Test basic ways of constructing interpolating splines.
    #
    def get_xy(self, xp):
        xx = xp.linspace(0., 2.*cupy.pi, 11)
        yy = xp.sin(xx)
        return xx, yy

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=TypeError)
    def test_non_int_order(self, xp, scp):
        x, y = self.get_xy(xp)
        return scp.interpolate.make_interp_spline(x, y, k=2.5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_order_0(self, xp, scp):
        x, y = self.get_xy(xp)
        return (scp.interpolate.make_interp_spline(x, y, k=0)(x),
                scp.interpolate.make_interp_spline(x, y, k=0, axis=-1)(x))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_linear(self, xp, scp):
        x, y = self.get_xy(xp)
        return (scp.interpolate.make_interp_spline(x, y, k=1)(x),
                scp.interpolate.make_interp_spline(x, y, k=1, axis=-1)(x))

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_incompatible_x_y(self, xp, scp, k):
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5, 6, 7]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x(self, xp, scp, k):
        x = [0, 1, 1, 2, 3, 4]      # duplicates
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x_2(self, xp, scp, k):
        x = [0, 2, 1, 3, 4, 5]      # unsorted
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x_3(self, xp, scp, k):
        x = xp.asarray([0, 1, 2, 3, 4, 5]).reshape((1, -1))     # 1D
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    @pytest.mark.parametrize('k', [3, 5])
    def test_not_a_knot(self, xp, scp, k):
        x, y = self.get_xy(xp)
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('k', [0, 1, 3, 5])
    def test_int_xy(self, xp, scp, k):
        x = xp.arange(10).astype(int)
        y = xp.arange(10).astype(int)
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_sliced_input(self, xp, scp, k):
        #  non C contiguous arrays
        xx = xp.linspace(-1, 1, 100)
        x = xx[::5]
        y = xx[::5]
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)


    """
    def test_quadratic_deriv(self):
        der = [(1, 8.)]  # order, value: f'(x) = 8.

        # derivative at right-hand edge
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(None, der))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[-1], 1), der[0][1], atol=1e-14, rtol=1e-14)

        # derivative at left-hand edge
        b = make_interp_spline(self.xx, self.yy, k=2, bc_type=(der, None))
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
        assert_allclose(b(self.xx[0], 1), der[0][1], atol=1e-14, rtol=1e-14)
    """

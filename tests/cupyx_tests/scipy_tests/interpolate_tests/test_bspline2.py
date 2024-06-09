import pytest
import cupy
from cupy import testing
from cupy.cuda import runtime

import numpy as _np
import cupyx.scipy.interpolate as csi  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
@testing.with_requires("scipy")
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

    @testing.with_requires("scipy >= 1.10")
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_incompatible_x_y(self, xp, scp, k):
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5, 6, 7]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.with_requires("scipy >= 1.10")
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x(self, xp, scp, k):
        x = [0, 1, 1, 2, 3, 4]      # duplicates
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.with_requires("scipy >= 1.10")
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x_2(self, xp, scp, k):
        x = [0, 2, 1, 3, 4, 5]      # unsorted
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.with_requires("scipy >= 1.10")
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x_3(self, xp, scp, k):
        x = xp.asarray([0, 1, 2, 3, 4, 5]).reshape((1, -1))     # 1D
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    @pytest.mark.parametrize('k', [3, 5])
    def test_not_a_knot(self, xp, scp, k):
        x, y = self.get_xy(xp)
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
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

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    @pytest.mark.parametrize('k', [1, 2, 3, 5])
    def test_list_input(self, xp, scp, k):
        # regression test for gh-8714: TypeError for x, y being lists and k=2
        x = list(range(10))
        y = [a**2 for a in x]
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_quadratic_deriv_right(self, xp, scp):
        x, y = self.get_xy(xp)
        der = [(1, 8.)]  # order, value: f'(x) = 8.

        # derivative at right-hand edge
        b = scp.interpolate.make_interp_spline(x, y, k=2, bc_type=(None, der))
        return b(x), b(x[-1], 1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_quadratic_deriv_left(self, xp, scp):
        x, y = self.get_xy(xp)
        der = [(1, 8.)]  # order, value: f'(x) = 8.

        # derivative at left-hand edge
        b = scp.interpolate.make_interp_spline(x, y, k=2, bc_type=(der, None))
        return b(x), b(x[0], 1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_cubic_deriv_deriv(self, xp, scp):
        x, y = self.get_xy(xp)
        # first derivatives at left & right edges:
        der_l, der_r = [(1, 3.)], [(1, 4.)]
        b = scp.interpolate.make_interp_spline(
            x, y, k=3, bc_type=(der_l, der_r))
        return b(x), b(x[0], 1), b(x[-1], 1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_cubic_deriv_natural(self, xp, scp):
        x, y = self.get_xy(xp)

        # 'natural' cubic spline, zero out 2nd derivatives at the boundaries
        der_l, der_r = [(2, 0)], [(2, 0)]
        b = scp.interpolate.make_interp_spline(x, y, k=3,
                                               bc_type=(der_l, der_r))
        return b(x), b(x[0], 2), b(x[-1], 2)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_quintic_derivs(self, xp, scp):
        k, n = 5, 7
        x = xp.arange(n).astype(xp.float64)
        y = xp.sin(x)
        der_l = [(1, -12.), (2, 1)]
        der_r = [(1, 8.), (2, 3.)]
        b = scp.interpolate.make_interp_spline(x, y, k=k,
                                               bc_type=(der_l, der_r))
        return b(x), b(x[0], 1), b(x[0], 2), b(x[-1], 1), b(x[-1], 2)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_knots_not_data_sites(self, xp, scp):
        # Knots need not coincide with the data sites.
        # use a quadratic spline, knots are at data averages,
        # two additional constraints are zero 2nd derivatives at edges
        k = 2
        x, y = self.get_xy(xp)
        t = xp.r_[(x[0],)*(k+1),
                  (x[1:] + x[:-1]) / 2.,
                  (x[-1],)*(k+1)]
        b = scp.interpolate.make_interp_spline(x, y, k, t,
                                               bc_type=([(2, 0)], [(2, 0)]))
        return b(x), b(x[0], 2), b(x[-1], 2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_minimum_points_and_deriv(self, xp, scp):
        # interpolation of f(x) = x**3 between 0 and 1. f'(x) = 3 * xx**2 and
        # f'(0) = 0, f'(1) = 3.
        k = 3
        x = [0., 1.]
        y = [0., 1.]
        b = scp.interpolate.make_interp_spline(x, y, k,
                                               bc_type=([(1, 0.)], [(1, 3.)]))
        xx = xp.linspace(0., 1.)
        return b(xx)    # assert_allclose(b(xx), xx**3)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_complex(self, xp, scp):
        x, y = self.get_xy(xp)
        y = y + 1j*y

        # first derivatives at left & right edges:
        der_l, der_r = [(1, 3.j)], [(1, 4.+2.j)]
        b = scp.interpolate.make_interp_spline(x, y, k=3,
                                               bc_type=(der_l, der_r))
        return b(x), b(x[0], 1), b(x[-1], 1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_01(self, xp, scp):
        # also test zero and first order
        x, y = self.get_xy(xp)
        y = y + 1j*y
        return (scp.interpolate.make_interp_spline(x, y, k=0)(x),
                scp.interpolate.make_interp_spline(x, y, k=1)(x))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_multiple_rhs(self, xp, scp):
        x, _ = self.get_xy(xp)
        yy = xp.c_[xp.sin(x), xp.cos(x)]
        der_l = [(1, [1., 2.])]
        der_r = [(1, [3., 4.])]

        b = scp.interpolate.make_interp_spline(x, yy, k=3,
                                               bc_type=(der_l, der_r))
        return b(x), b(x[0], 1), b(x[-1], 1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes(self, xp, scp):
        xp.random.seed(1234)
        k, n = 3, 22
        x = xp.sort(xp.random.random(size=n))
        y = xp.random.random(size=(n, 5, 6, 7))

        b1 = scp.interpolate.make_interp_spline(x, y, k)
        assert b1.c.shape == (n, 5, 6, 7)

        # now throw in some derivatives
        d_l = [(1, xp.random.random((5, 6, 7)))]
        d_r = [(1, xp.random.random((5, 6, 7)))]
        b2 = scp.interpolate.make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        assert b2.c.shape == (n + k - 1, 5, 6, 7)

        return b1.c.shape + b2.c.shape

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_1(self, xp, scp):
        x, _ = self.get_xy(xp)
        y = xp.sin(x)

        # a single string is duplicated
        b1 = scp.interpolate.make_interp_spline(x, y, k=3,
                                                bc_type='natural')
        b2 = scp.interpolate.make_interp_spline(x, y, k=3,
                                                bc_type=([(2, 0)], [(2, 0)]))
        return b1.c, b2.c

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_2(self, xp, scp):
        x, _ = self.get_xy(xp)
        y = xp.sin(x)
        # two strings are handled
        b1 = scp.interpolate.make_interp_spline(x, y, k=3,
                                                bc_type=('natural', 'clamped'))
        b2 = scp.interpolate.make_interp_spline(x, y, k=3,
                                                bc_type=([(2, 0)], [(1, 0)]))
        return b1.c, b2.c

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_3(self, xp, scp):
        x, _ = self.get_xy(xp)
        y = xp.sin(x)

        # one-sided BCs are OK
        b1 = scp.interpolate.make_interp_spline(x, y, k=2,
                                                bc_type=(None, 'clamped'))
        b2 = scp.interpolate.make_interp_spline(x, y, k=2,
                                                bc_type=(None, [(1, 0.0)]))
        return b1.c, b2.c

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_4(self, xp, scp):
        x, _ = self.get_xy(xp)
        y = xp.sin(x)

        # 'not-a-knot' is equivalent to None
        b1 = scp.interpolate.make_interp_spline(x, y, k=3,
                                                bc_type='not-a-knot')
        b2 = scp.interpolate.make_interp_spline(x, y, k=3,
                                                bc_type=None)
        return b1.c, b2.c

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_string_aliases_5(self, xp, scp):
        x, _ = self.get_xy(xp)
        y = xp.sin(x)

        # unknown strings do not pass
        scp.interpolate.make_interp_spline(x, y, k=3, bc_type='typo')

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_6(self, xp, scp):
        x, _ = self.get_xy(xp)

        # string aliases are handled for 2D values
        yy = xp.c_[xp.sin(x), xp.cos(x)]
        der_l = [(1, [0., 0.])]
        der_r = [(2, [0., 0.])]

        b2 = scp.interpolate.make_interp_spline(x, yy, k=3,
                                                bc_type=(der_l, der_r))
        b1 = scp.interpolate.make_interp_spline(x, yy, k=3,
                                                bc_type=('clamped', 'natural'))
        return b1.c, b2.c

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_string_aliases_7(self, xp, scp):
        # ... and for N-D values:

        # cupy and numpy random streams differ for the same seed,
        # hence use numpy.random for both numpy and cupy
        rng = _np.random.RandomState(1234)
        k, n = 3, 22
        x = _np.sort(rng.uniform(size=n))
        y = rng.uniform(size=(n, 5, 6, 7))
        if xp is cupy:
            x = cupy.asarray(x)
            y = cupy.asarray(y)

        # now throw in some derivatives
        d_l = [(1, xp.zeros((5, 6, 7)))]
        d_r = [(1, xp.zeros((5, 6, 7)))]
        b1 = scp.interpolate.make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        b2 = scp.interpolate.make_interp_spline(x, y, k, bc_type='clamped')
        return b1.c, b2.c

    def test_deriv_spec(self):
        # If one of the derivatives is omitted, the spline definition is
        # incomplete.
        x = y = [1.0, 2, 3, 4, 5, 6]

        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=([(1, 0.)], None))

        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=(1, 0.))

        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=[(1, 0.)])

        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=42)

        # CubicSpline expects`bc_type=(left_pair, right_pair)`, while
        # here we expect `bc_type=(iterable, iterable)`.
        l, r = (1, 0.0), (1, 0.0)
        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=(l, r))

    def test_full_matrix(self):
        from cupyx.scipy.interpolate._bspline2 import (
            _make_interp_spline_full_matrix)
        cupy.random.seed(1234)
        k, n = 3, 7
        x = cupy.sort(cupy.random.random(size=n))
        y = cupy.random.random(size=n)

        # test not-a-knot
        b = csi.make_interp_spline(x, y, k=3)
        bf = _make_interp_spline_full_matrix(x, y, k, b.t, bc_type=None)
        cupy.testing.assert_allclose(b.c, bf.c, atol=1e-14, rtol=1e-14)

        # test with some b.c.
        b = csi.make_interp_spline(x, y, k=3, bc_type='natural')
        bf = _make_interp_spline_full_matrix(x, y, k, b.t, bc_type='natural')
        cupy.testing.assert_allclose(b.c, bf.c, atol=1e-13)

    # test periodic constructor #


@testing.with_requires("scipy>=1.7")
@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
class TestInterpPeriodic:
    #
    # Test basic ways of constructing interpolating splines w/periodic BCs.
    #
    def get_xy(self, xp):
        xx = xp.linspace(0., 2.*cupy.pi, 11)
        yy = xp.sin(xx)
        return xx, yy

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic(self, xp, scp):
        x, y = self.get_xy(xp)
        # k = 5 here for more derivatives
        b = scp.interpolate.make_interp_spline(x, y, k=5, bc_type='periodic')
        for i in range(1, 5):
            xp.testing.assert_allclose(b(x[0], nu=i),
                                       b(x[-1], nu=i), atol=1e-11)
        return b(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic_axis1(self, xp, scp):
        x, y = self.get_xy(xp)
        b = scp.interpolate.make_interp_spline(x, y, k=5,
                                               bc_type='periodic', axis=-1)
        for i in range(1, 5):
            xp.testing.assert_allclose(b(x[0], nu=i),
                                       b(x[-1], nu=i), atol=1e-11)
        return b(x)

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic_random(self, xp, scp, k):
        # tests for both cases (k > n and k <= n)
        n = 15
        _np.random.seed(1234)
        x = _np.sort(_np.random.random_sample(n) * 10)
        y = _np.random.random_sample(n) * 100
        if xp is cupy:
            x = cupy.asarray(x)
            y = cupy.asarray(y)

        y[0] = y[-1]
        b = scp.interpolate.make_interp_spline(x, y, k=k, bc_type='periodic')

        return b(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic_axis(self, xp, scp):
        x, y = self.get_xy(xp)
        n = x.shape[0]
        _np.random.seed(1234)
        x = _np.random.random_sample(n) * 2 * _np.pi
        x = _np.sort(x)
        if xp is cupy:
            x = cupy.asarray(x)
        x[0] = 0.
        x[-1] = 2 * xp.pi
        y = xp.zeros((2, n))
        y[0] = xp.sin(x)
        y[1] = xp.cos(x)
        b = scp.interpolate.make_interp_spline(x, y, k=5,
                                               bc_type='periodic', axis=1)
        return b(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_periodic_points_exception(self, xp, scp):
        # first and last points should match when periodic case expected
        n, k = 8, 5
        x = xp.linspace(0, n, n)
        y = x
        return scp.interpolate.make_interp_spline(x, y, k=k,
                                                  bc_type='periodic')

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_periodic_knots_exception(self, xp, scp):
        # `periodic` case does not work with passed vector of knots
        n, k = 7, 3
        x = xp.linspace(0, n, n)
        y = x**2
        t = xp.zeros(n + 2 * k)
        return scp.interpolate.make_interp_spline(x, y, k, t, 'periodic')

    def test_periodic_cubic(self):
        # edge case: Cubic interpolation on 3 points
        # TODO: refactor when PPoly / CubicHermiteSpline is available
        n = 3
        cupy.random.seed(1234)
        x = cupy.sort(cupy.random.random_sample(n) * 10)
        y = cupy.random.random_sample(n) * 100
        y[0] = y[-1]
        b = csi.make_interp_spline(x, y, k=3, bc_type='periodic')

        cub = interpolate.CubicSpline(x.get(), y.get(), bc_type='periodic')
        cupy.testing.assert_allclose(b(x),
                                     cub(x.get()), atol=1e-14)

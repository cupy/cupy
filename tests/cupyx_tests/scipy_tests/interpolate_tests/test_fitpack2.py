import pytest
import cupy
from cupy import testing

import cupyx.scipy.interpolate as csi  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@testing.with_requires("scipy")
class TestUnivariateSpline:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_linear_constant(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([3, 3, 3])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        return lut.get_knots(), lut.get_coeffs(), lut(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_preserve_shape(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([0, 2, 4])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        arg = 2
        return lut(arg), lut(arg, nu=1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_preserve_shape_2(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([0, 2, 4])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        arg = xp.asarray([1.5, 2, 2.5])
        return lut(arg), lut(arg, nu=1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_linear_1d(self, xp, scp):
        x = xp.asarray([1, 2, 3])
        y = xp.asarray([0, 2, 4])
        lut = scp.interpolate.UnivariateSpline(x, y, k=1)
        return lut.get_knots(), lut.get_coeffs(), lut(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty_input(self, xp, scp):
        # Test whether empty input returns an empty output. Ticket 1014
        x = xp.asarray([1, 3, 5, 7, 9])
        y = xp.asarray([0, 4, 9, 12, 21])
        spl = scp.interpolate.UnivariateSpline(x, y, k=3)
        return spl([])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivatives(self, xp, scp):
        x = xp.asarray([1, 3, 5, 7, 9])
        y = xp.asarray([0, 4, 9, 12, 21])
        spl = scp.interpolate.UnivariateSpline(x, y, k=3)
        return spl.derivatives(3.5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivatives_2(self, xp, scp):
        x = xp.arange(8)
        y = x**3 + 2.*x**2

        spl = scp.interpolate.UnivariateSpline(x, y, s=0, k=3)
        return spl.derivatives(3)

    @pytest.mark.parametrize('klass',
                             ['UnivariateSpline',
                              'InterpolatedUnivariateSpline']
                             )
    @pytest.mark.parametrize('ext', ['extrapolate', 'zeros', 'const'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_out_of_range_regression(self, klass, ext, xp, scp):
        x = xp.arange(5, dtype=float)
        y = x**3

        xp = xp.linspace(-8, 13, 100)
        cls = getattr(scp.interpolate, klass)
        spl = cls(x=x, y=y)
        return spl(xp, ext=ext)

    def test_lsq_fpchec(self):
        xs = cupy.arange(100) * 1.
        ys = cupy.arange(100) * 1.
        knots = cupy.linspace(0, 99, 10)
        bbox = (-1, 101)
        with pytest.raises(ValueError):
            csi.LSQUnivariateSpline(xs, ys, knots, bbox=bbox)

    @pytest.mark.parametrize('ext', [0, 1, 2, 3])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integral_out_of_bounds(self, xp, scp, ext):
        # Regression test for gh-7906: .integral(a, b) is wrong if both
        # a and b are out-of-bounds
        x = xp.linspace(0., 1., 7)
        f = scp.interpolate.UnivariateSpline(x, x, s=0, ext=ext)
        vals = [f.integral(a, b)
                for (a, b) in [(1, 1), (1, 5), (2, 5),
                               (0, 0), (-2, 0), (-2, -1)]
                ]
        # NB: scipy returns python floats, cupy returns 0D arrays
        return xp.asarray(vals)

    @pytest.mark.parametrize('s', [0, 0.1, 0.01])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_values(self, xp, scp, s):
        x = xp.arange(8) + 0.5
        y = x + 1 / (1 - x)
        spl = scp.interpolate.UnivariateSpline(x, y, s=s)
        return spl(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_set_smoothing_factor(self, xp, scp):
        x = xp.arange(8) + 0.5
        y = x + 1 / (1 - x)
        spl = scp.interpolate.UnivariateSpline(x, y, s=0.1)
        spl.set_smoothing_factor(s=0.05)
        return spl(x)

    def test_reset_class(self):
        # SciPy weirdness: *UnivariateSpline.__class__ may change
        x = cupy.arange(8) + 0.5
        y = x + 1 / (1 - x)
        spl = csi.UnivariateSpline(x, y, s=0.1)

        assert spl.__class__ == csi.UnivariateSpline

        spl.set_smoothing_factor(s=0)
        assert spl.__class__ == csi.InterpolatedUnivariateSpline

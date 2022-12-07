import pytest
import numpy

import cupy
from cupy import testing
import cupyx.scipy.interpolate  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass

# TODO: add BPoly, if/when implemented
parametrize_cls = pytest.mark.parametrize('cls', ['PPoly'])


@testing.with_requires("scipy")
class TestPPolyCommon:
    """Test basic functionality for PPoly and BPoly."""

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_sort_check(self, xp, scp, cls):
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 1, 0.5])
        cls = getattr(scp.interpolate, cls)
        cls(c, x)

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_ctor_c(self, xp, scp, cls):
        # wrong shape: `c` must be at least 2D
        cls = getattr(scp.interpolate, cls)
        cls(c=xp.asarray([1, 2]),
            x=xp.asarray([0, 1]))

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extend(self, xp, scp, cls):
        # Test adding new points to the piecewise polynomial
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)

        order = 3
        x = numpy.unique(numpy.r_[0, 10 * numpy.random.rand(30), 10])
        x = xp.asarray(x)
        c = 2*numpy.random.rand(order+1, len(x)-1, 2, 3) - 1
        c = xp.asarray(c)

        pp = cls(c[:,:9], x[:10])
        pp.extend(c[:,9:], x[10:])

        pp2 = cls(c[:, 10:], x[10:])
        pp2.extend(c[:, :10], x[:10])

        pp3 = cls(c, x)

        return pp.c, pp.x, pp2.c, pp2.x, pp3.c, pp3.x

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extend_diff_orders(self, xp, scp, cls):
        # Test extending polynomial with different order one
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)

        x = xp.linspace(0, 1, 6)
        c = xp.asarray(numpy.random.rand(2, 5))

        x2 = xp.linspace(1, 2, 6)
        c2 = xp.asarray(numpy.random.rand(4, 5))

        pp1 = cls(c, x)
        pp2 = cls(c2, x2)

        pp_comb = cls(c, x)
        pp_comb.extend(c2, x2[1:])

        # NB. doesn't match to pp1 at the endpoint, because pp1 is not
        #     continuous with pp2 as we took random coefs.
        xi1 = xp.linspace(0, 1, 300, endpoint=False)
        xi2 = xp.linspace(1, 2, 300)

        return pp1(xi1), pp_comb(xi1), pp2(xi2), pp_comb(xi2)

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extend_descending(self, xp, scp, cls):
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(0)

        order = 3
        x = numpy.sort(numpy.random.uniform(0, 10, 20))
        x = xp.asarray(x)
        c = numpy.random.rand(order + 1, x.shape[0] - 1, 2, 3)
        c = xp.asarray(c)

        p = cls(c, x)

        p1 = cls(c[:, :9], x[:10])
        p1.extend(c[:, 9:], x[10:])

        p2 = cls(c[:, 10:], x[10:])
        p2.extend(c[:, :10], x[:10])

        return p1.c, p1.x, p2.c, p2.x

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shape(self, xp, scp, cls):
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)

        c = numpy.random.rand(8, 12, 5, 6, 7)
        c = xp.asarray(c)
        x = numpy.sort(numpy.random.rand(13))
        x = xp.asarray(x)
        xpts = numpy.random.rand(3, 4)
        xpts = xp.asarray(xpts)

        p = cls(c, x)
        return p(xpts).shape

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_shape_2(self, xp, scp, cls):
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)

        c = numpy.random.rand(8, 12, 5, 6, 7)
        c = xp.asarray(c)
        x = numpy.sort(numpy.random.rand(13))
        x = xp.asarray(x)

        # 'scalars'
        p = cls(c[..., 0, 0, 0], x)

        assert p(0.5).shape == ()
        assert p(xp.array(0.5)).shape == ()

        xxx = xp.array([[0.1, 0.2], [0.4]], dtype=object)
        p(xxx)   # raises ValueError

    @parametrize_cls
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_coef(self, xp, scp, cls):
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(12345)
        x = numpy.sort(numpy.random.random(13))
        x = xp.array(x)
        c = numpy.random.random((8, 12)) * (1. + 0.3j)
        c = xp.array(c)
        c_re, c_im = c.real, c.imag
        xpt = xp.array(numpy.random.random(5))

        p = cls(c, x)
        return [p(xpt, nu) for nu in [0, 1, 2]]        

    @parametrize_cls
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_axis(self, xp, scp, cls, axis):
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(12345)
        c = numpy.random.rand(3, 4, 5, 6, 7, 8)
        c = xp.asarray(c)
        c_s = c.shape
        xpt = numpy.random.random((1, 2))
        xpt = xp.asarray(xpt)

        m = c.shape[axis+1]
        x = numpy.sort(numpy.random.rand(m+1))
        x = xp.asarray(x)

        p = cls(c, x, axis=axis)
        assert (p.c.shape ==
                c_s[axis:axis+2] + c_s[:axis] + c_s[axis+2:])

        res = p(xpt)
        targ_shape = c_s[:axis] + xpt.shape + c_s[2+axis:]
        assert res.shape == targ_shape

        # deriv/antideriv does not drop the axis
        for p1 in [cls(c, x, axis=axis).derivative(),
                   cls(c, x, axis=axis).derivative(2),
                   cls(c, x, axis=axis).antiderivative(),
                   cls(c, x, axis=axis).antiderivative(2)]:
            assert p1.axis == p.axis

        return True

    @parametrize_cls
    @pytest.mark.parametrize('axis', [-1, 4, 5, 6])
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_axis_2(self, xp, scp, cls, axis):
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(12345)
        c = numpy.random.rand(3, 4, 5, 6, 7, 8)
        c = xp.asarray(c)

        x = numpy.sort(numpy.random.rand(c.shape[-1]))
        x = xp.asarray(x)

        # c array needs two axes for the coefficients and intervals, so
        # 0 <= axis < c.ndim-1; raise otherwise
        cls(c=c, x=x, axis=axis)

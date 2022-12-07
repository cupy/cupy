import pytest
import numpy
import numpy as np

import cupy
from cupy import testing
from cupy.testing import assert_allclose
import cupyx.scipy.interpolate  # NOQA

try:
    from scipy import interpolate as sc_interpolate  # NOQA
    from scipy import special as sc_special   # NOQA
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

        pp = cls(c[:, :9], x[:10])
        pp.extend(c[:, 9:], x[10:])

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
        # c_re, c_im = c.real, c.imag
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
        instance = cls(c=c, x=x, axis=axis)
        return instance.c, instance.x, instance.axis


@testing.with_requires("scipy")
class TestPPoly:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple(self, xp, scp):
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly(c, x)
        testing.assert_allclose(p(0.3), 1*0.3**2 + 2*0.3 + 3)
        testing.assert_allclose(p(0.7), 4*(0.7-0.5)**2 + 5*(0.7-0.5) + 6)
        return True

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_periodic(self, xp, scp):
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly(c, x, extrapolate='periodic')

        testing.assert_allclose(p(1.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
        testing.assert_allclose(p(-0.3), 4 * (0.7 - 0.5)
                                ** 2 + 5 * (0.7 - 0.5) + 6)

        testing.assert_allclose(p(1.3, 1), 2 * 0.3 + 2)
        testing.assert_allclose(p(-0.3, 1), 8 * (0.7 - 0.5) + 5)
        return True

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_read_only(self, xp, scp):
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 0.5, 1])
        xnew = xp.array([0, 0.1, 0.2])
        scp.interpolate.PPoly(c, x, extrapolate='periodic')

        lst = []
        for writeable in (True, False):
            x.flags.writeable = writeable
            f = scp.interpolate.PPoly(c, x)
            vals = f(xnew)
            assert xp.isfinite(vals).all()
            lst.append(vals)
        return vals

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multi_shape(self, xp, scp):
        c = numpy.random.rand(6, 2, 1, 2, 3)
        c = xp.asarray(c)
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly(c, x)
        assert p.x.shape == x.shape
        assert p.c.shape == c.shape
        assert p(0.3).shape == c.shape[2:]

        assert p(xp.random.rand(5, 6)).shape == (5, 6) + c.shape[2:]

        dp = p.derivative()
        assert dp.c.shape == (5, 2, 1, 2, 3)
        ip = p.antiderivative()
        assert ip.c.shape == (7, 2, 1, 2, 3)
        return True

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_construct_fast(self, xp, scp):
        c = xp.array([[1, 4], [2, 5], [3, 6]], dtype=float)
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly.construct_fast(c, x)
        testing.assert_allclose(p(0.3), 1*0.3**2 + 2*0.3 + 3)
        testing.assert_allclose(p(0.7), 4*(0.7-0.5)**2 + 5*(0.7-0.5) + 6)
        return p(0.3), p(0.7)

    def test_from_spline(self):
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))

        spl = sc_interpolate.splrep(x, y, s=0)
        spl = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])

        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl)

        xi = np.linspace(0, 1, 200)
        testing.assert_allclose(pp(xi), sc_interpolate.splev(xi, spl))

        # make sure .from_spline accepts BSpline objects
        b = cupyx.scipy.interpolate.BSpline(*spl)
        ppp = cupyx.scipy.interpolate.PPoly.from_spline(b)
        testing.assert_allclose(ppp(xi), b(xi))

        # BSpline's extrapolate attribute propagates unless overridden
        t, c, k = spl
        for extrap in (None, True, False):
            b = cupyx.scipy.interpolate.BSpline(t, c, k, extrapolate=extrap)
            p = cupyx.scipy.interpolate.BSplinePPoly.from_spline(b)
            assert p.extrapolate == b.extrapolate

    def test_derivative_simple(self):
        c = cupy.array([[4, 3, 2, 1]]).T
        dc = cupy.array([[3*4, 2*3, 2]]).T
        ddc = cupy.array([[2*3*4, 1*2*3]]).T
        x = cupy.array([0, 1])

        pp = cupyx.scipy.interpolate.PPoly(c, x)
        dpp = cupyx.scipy.interpolate.PPoly(dc, x)
        ddpp = cupyx.scipy.interpolate.PPoly(ddc, x)

        testing.assert_allclose(pp.derivative().c, dpp.c)
        testing.assert_allclose(pp.derivative(2).c, ddpp.c)

    def test_derivative_eval(self):
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))

        spl = sc_interpolate.splrep(x, y, s=0)
        spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl_cupy)

        xi = cupy.linspace(0, 1, 200)
        for dx in range(0, 3):
            testing.assert_allclose(pp(xi, dx),
                                    sc_interpolate.splev(xi, spl, dx))

    def test_derivative(self):
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))

        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl_cupy)

        xi = cupy.linspace(0, 1, 200)
        for dx in range(0, 10):
            testing.assert_allclose(pp(xi, dx), pp.derivative(dx)(xi),
                                    err_msg="dx=%d" % (dx,))

    def test_antiderivative_of_constant(self):
        # https://github.com/scipy/scipy/issues/4216
        PPoly = cupyx.scipy.interpolate.PPoly
        p = PPoly(cupy.asarray([[1.]]), cupy.asarray([0, 1]))
        testing.assert_allclose(p.antiderivative().c,
                                PPoly(c=cupy.asarray([[1], [0]]),
                                      x=cupy.asarray([0, 1])).c, atol=1e-15)
        testing.assert_allclose(p.antiderivative().x,
                                PPoly(c=cupy.asarray([[1], [0]]),
                                      x=cupy.asarray([0, 1])).x, atol=1e-15)

    def test_antiderivative_regression_4355(self):
        # https://github.com/scipy/scipy/issues/4355
        PPoly = cupyx.scipy.interpolate.PPoly
        p = PPoly(cupy.asarray([[1., 0.5]]), cupy.asarray([0, 1, 2]))
        q = p.antiderivative()
        testing.assert_allclose(q.c, cupy.asarray([[1, 0.5], [0, 1]]))
        testing.assert_allclose(q.x, cupy.asarray([0, 1, 2]))
        testing.assert_allclose(p.integrate(0, 2), 1.5)
        testing.assert_allclose(q(2) - q(0), 1.5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antiderivative_simple(self, xp, scp):
        # [ p1(x) = 3*x**2 + 2*x + 1,
        #   p2(x) = 1.6875]
        c = xp.array([[3, 2, 1], [0, 0, 1.6875]]).T
        # [ pp1(x) = x**3 + x**2 + x,
        #   pp2(x) = 1.6875*(x - 0.25) + pp1(0.25)]
        # ic = xp.array([[1, 1, 1, 0], [0, 0, 1.6875, 0.328125]]).T
        # [ ppp1(x) = (1/4)*x**4 + (1/3)*x**3 + (1/2)*x**2,
        #   ppp2(x) = (1.6875/2)*(x - 0.25)**2 + pp1(0.25)*x + ppp1(0.25)]
        # iic = xp.array([[1/4, 1/3, 1/2, 0, 0],
        #                 [0, 0, 1.6875/2, 0.328125, 0.037434895833333336]]).T
        x = xp.array([0, 0.25, 1])

        pp = scp.interpolate.PPoly(c, x)
        ipp = pp.antiderivative()
        iipp = pp.antiderivative(2)
        iipp2 = ipp.antiderivative()

        return ipp.x, ipp.c, iipp, iipp2

    def test_antiderivative_vs_derivative(self):
        numpy.random.seed(1234)
        x = numpy.linspace(0, 1, 30)**2
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        spl = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl)

        for dx in range(0, 10):
            ipp = pp.antiderivative(dx)

            # check that derivative is inverse op
            pp2 = ipp.derivative(dx)
            assert_allclose(pp.c, pp2.c)

            # check continuity
            for k in range(dx):
                pp2 = ipp.derivative(k)

                r = 1e-13
                endpoint = r*pp2.x[:-1] + (1 - r)*pp2.x[1:]

                assert_allclose(pp2(pp2.x[1:]), pp2(endpoint),
                                rtol=1e-7, err_msg="dx=%d k=%d" % (dx, k))

    def test_antiderivative_vs_spline(self):
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))

        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl)

        for dx in range(0, 10):
            pp2 = pp.antiderivative(dx)
            spl2 = cupyx.scipy.interpolatesplantider(spl_cupy, dx)
            spl2_np = (spl2[0].get(), spl2[1].get(), spl2[2])

            xi = cupy.linspace(0, 1, 200)
            testing.assert_allclose(pp2(xi),
                                    sc_interpolate.splev(xi.get(), spl2_np),
                                    rtol=1e-7)

    def test_antiderivative_continuity(self):
        c = cupy.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
        x = cupy.array([0, 0.5, 1])

        p = cupyx.scipy.interpolate.PPoly(c, x)
        ip = p.antiderivative()

        # check continuity
        testing.assert_allclose(ip(0.5 - 1e-9), ip(0.5 + 1e-9), rtol=1e-8)

        # check that only lowest order coefficients were changed
        p2 = ip.derivative()
        testing.assert_allclose(p2.c, p.c)

    def test_integrate(self):
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))

        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        # spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl)

        a, b = 0.3, 0.9
        ig = pp.integrate(a, b)

        ipp = pp.antiderivative()
        testing.assert_allclose(ig, ipp(b) - ipp(a))
        testing.assert_allclose(ig, sc_interpolate.splint(a, b, spl))

        a, b = -0.3, 0.9
        ig = pp.integrate(a, b, extrapolate=True)
        testing.assert_allclose(ig, ipp(b) - ipp(a))

        assert (cupy.isnan(pp.integrate(a, b, extrapolate=False)).all())

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate_readonly(self, xp, scp):
        x = xp.array([1, 2, 4])
        c = xp.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])

        ret = []
        for writeable in (True, False):
            x.flags.writeable = writeable

            P = scp.interpolate.PPoly(c, x)
            vals = P.integrate(1, 4)
            ret.append(vals)

        return ret

    def test_integrate_periodic(self):
        x = cupy.array([1, 2, 4])
        c = cupy.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])

        P = cupyx.scipy.interpolate.PPoly(c, x, extrapolate='periodic')
        poly_int = P.antiderivative()

        period_int = poly_int(4) - poly_int(1)

        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)

        assert_allclose(P.integrate(1.5, 2.5), poly_int(2.5) - poly_int(1.5))
        assert_allclose(
            P.integrate(3.5, 5), poly_int(2) - poly_int(1) +
            poly_int(4) - poly_int(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12),
                        poly_int(2) - poly_int(1) + poly_int(4) -
                        poly_int(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12),
                        poly_int(2) - poly_int(1) + poly_int(4) -
                        poly_int(3.5) + 4 * period_int)

        assert_allclose(P.integrate(0, -1), poly_int(2) - poly_int(3))
        assert_allclose(P.integrate(-9, -10), poly_int(2) - poly_int(3))
        assert_allclose(
            P.integrate(0, -10), poly_int(2) - poly_int(3) - 3 * period_int)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_roots(self, xp, scp):
        x = xp.linspace(0, 1, 31)**2
        y = xp.sin(30*x)

        if xp is cupy:
            spl = sc_interpolate.splrep(x.get(), y.get(), s=0, k=3)
            spl = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        else:
            spl = sc_interpolate.splrep(x, y, s=0, k=3)

        pp = scp.interpolate.PPoly.from_spline(spl)

        r = pp.roots()
        return r

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_roots_idzero(self, xp, scp):
        # Roots for piecewise polynomials with identically zero
        # sections.
        c = xp.array([[-1, 0.25], [0, 0], [-1, 0.25]]).T
        x = xp.array([0, 0.4, 0.6, 1.0])
        pp = scp.interpolate.PPoly(c, x)

        # ditto for p.solve(const) with sections identically equal const
        const = 2.
        c1 = c.copy()
        c1[1, :] += const
        pp1 = scp.interpolate.PPoly(c1, x)

        return pp.roots(), pp1.roots()

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_all_zero(self, xp, scp):
        # test the code path for the polynomial being
        # identically zero everywhere
        c = xp.asarray([[0], [0]])
        x = xp.asarray([0, 1])
        p = scp.interpolate.PPoly(c, x)
        return p.roots(), p.solve(0), p.solve(1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_all_zero_1(self, xp, scp):
        # test the code path for the polynomial being
        # identically zero everywhere
        c = xp.asarray([[0, 0], [0, 0]])
        x = xp.asarray([0, 1, 2])
        p = scp.interpolate.PPoly(c, x)
        return p.roots(), p.solve(0), p.solve(1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_repeated(self, xp, scp):
        # Check roots repeated in multiple sections are reported only
        # once.

        # [(x + 1)**2 - 1, -x**2] ; x == 0 is a repeated root
        c = xp.array([[1, 0, -1], [-1, 0, 0]]).T
        x = xp.array([-1, 0, 1])

        pp = scp.interpolate.PPoly(c, x)
        return pp.roots(), pp.roots(extrapolate=False)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_discont(self, xp, scp):
        # Check that a discontinuity across zero is reported as root
        c = xp.array([[1], [-1]]).T
        x = xp.array([0, 0.5, 1])
        pp = scp.interpolate.PPoly(c, x)
        return (pp.roots(), pp.roots(discontinuity=False),
                pp.solve(0.5), pp.solve(0.5, discontinuity=False),
                pp.solve(1.5), pp.solve(1.5, discontinuity=False))

    def test_roots_random(self):
        # Check high-order polynomials with random coefficients
        numpy.random.seed(1234)

        num = 0

        for extrapolate in (True, False):
            for order in range(0, 20):
                x = numpy.unique(numpy.r_[0, 10 * numpy.random.rand(30), 10])
                x = cupy.asarray(x)
                c = 2*numpy.random.rand(order+1, len(x)-1, 2, 3) - 1
                c = cupy.asarray(c)

                pp = cupyx.scipy.interpolate.PPoly(c, x)
                for y in [0, numpy.random.random()]:
                    r = pp.solve(y, discontinuity=False,
                                 extrapolate=extrapolate)

                    for i in range(2):
                        for j in range(3):
                            rr = r[i, j]
                            if rr.size > 0:
                                # Check that the reported roots
                                # indeed are roots
                                num += rr.size
                                val = pp(rr, extrapolate=extrapolate)[:, i, j]
                                cmpval = pp(rr, nu=1,
                                            extrapolate=extrapolate)[:, i, j]
                                msg = "(%r) r = %s" % (extrapolate, repr(rr),)
                                assert_allclose((val-y) / cmpval, 0, atol=1e-7,
                                                err_msg=msg)

        # Check that we checked a number of roots
        assert num > 100, repr(num)

    # XXX: expose _croot_poly1 or skip
    '''
    def test_roots_croots(self):
        # Test the complex root finding algorithm
        np.random.seed(1234)

        for k in range(1, 15):
            c = np.random.rand(k, 1, 130)

            if k == 3:
                # add a case with zero discriminant
                c[:,0,0] = 1, 2, 1

            for y in [0, np.random.random()]:
                w = np.empty(c.shape, dtype=complex)
                _ppoly._croots_poly1(c, w)

                if k == 1:
                    assert_(np.isnan(w).all())
                    continue

                res = 0
                cres = 0
                for i in range(k):
                    res += c[i,None] * w**(k-1-i)
                    cres += abs(c[i,None] * w**(k-1-i))
                with np.errstate(invalid='ignore'):
                    res /= cres
                res = res.ravel()
                res = res[~np.isnan(res)]
                assert_allclose(res, 0, atol=1e-10)
    '''

    @pytest.mark.parametrize('extrapolate', [True, False, None])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extrapolate_attr(self, xp, scp, extrapolate):
        # [ 1 - x**2 ]
        c = xp.array([[-1, 0, 1]]).T
        x = xp.array([0, 1])

        pp = scp.interpolate.PPoly(c, x, extrapolate=extrapolate)
        pp_d = pp.derivative()
        pp_i = pp.antiderivative()

        xx = xp.asarray([-0.1, 1.1])

        return pp(xx), pp_i(xx), pp_d(xx), pp.roots()

    def binom_matrix(self, power, xp):
        n = numpy.arange(power + 1).reshape(-1, 1)
        k = numpy.arange(power + 1)
        B = sc_special.binom(n, k)
        if xp is cupy:
            B = cupy.asarray(B)
        return B[::-1, ::-1]

    def _prepare_descending(self, m, xp, scp):
        power = 3
        x = numpy.sort(numpy.random.uniform(0, 10, m + 1))
        x = xp.asarray(x)
        ca = numpy.random.uniform(-2, 2, size=(power + 1, m))
        ca = xp.asarray(ca)

        h = xp.diff(x)
        h_powers = h[None, :] ** xp.arange(power + 1)[::-1, None]
        B = self.binom_matrix(power, xp)
        cap = ca * h_powers
        cdp = xp.dot(B.T, cap)
        cd = cdp / h_powers

        pa = scp.interpolate.PPoly(ca, x, extrapolate=True)
        pd = scp.interpolate.PPoly(cd[:, ::-1], x[::-1], extrapolate=True)
        return pa, pd

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13)
    def test_descending(self, m, xp, scp):
        numpy.random.seed(0)
        pa, pd = self._prepare_descending(m, xp, scp)

        x_test = numpy.random.uniform(-10, 20, 100)
        x_test = xp.asarray(x_test)
        return pa(x_test), pa(x_test, 1)

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13)
    def test_descending_derivative(self, m, xp, scp):
        numpy.random.seed(0)
        pa, pd = self._prepare_descending(m, xp, scp)
        pa_d = pa.derivative()
        pd_d = pd.derivative()

        x_test = numpy.random.uniform(-10, 20, 100)
        x_test = xp.asarray(x_test)
        return pa_d(x_test), pd_d(x_test)

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_descending_antiderivative(self, m, xp, scp):
        numpy.random.seed(0)
        pa, pd = self._prepare_descending(m, xp, scp)

        # Antiderivatives won't be equal because fixing continuity is
        # done in the reverse order, but surely the differences should be
        # equal.
        pa_i = pa.antiderivative()
        pd_i = pd.antiderivative()
        for a, b in numpy.random.uniform(-10, 20, (5, 2)):
            int_a = pa.integrate(a, b)
            int_d = pd.integrate(a, b)
            testing.assert_allclose(int_a, int_d, rtol=1e-13)
            testing.assert_allclose(pa_i(b) - pa_i(a), pd_i(b) - pd_i(a),
                                    rtol=1e-13)
        return True

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-12)
    def test_descending_roots(self, m, xp, scp):
        numpy.random.seed(0)
        pa, pd = self._prepare_descending(m, xp, scp)

        roots_d = pd.roots()
        roots_a = pa.roots()
        return roots_a, roots_d

import pytest
import inspect

from cupy import testing
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import numpy as np
import cupyx.scipy.interpolate  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'extrapolate': [True, False, 'periodic'],
    'c': [[-1, 2, 0, -1], [[-1, 2, 0, -1]] * 5]}))
@testing.with_requires("scipy")
class TestBSpline:

    def _make_random_spline(self, xp, scp, n=35, k=3):
        t = xp.sort(testing.shaped_random((n + k + 1,), xp, dtype=np.float64))
        c = testing.shaped_random((n,), xp, dtype=np.float64)
        return scp.interpolate.BSpline.construct_fast(t, c, k)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=True)
    def test_ctor(self, xp, scp):
        # knots should be an ordered 1-D array of finite real numbers
        t = [1, 1.j]
        c = [1.0]
        k = 0
        scp.interpolate.BSpline(t, c, k)

        t = [1, xp.nan]
        scp.interpolate.BSpline(t, c, k)

        t = [1, xp.inf]
        scp.interpolate.BSpline(t, c, k)

        t = [1, -1]
        scp.interpolate.BSpline(t, c, k)

        t = [[1], [1]]
        scp.interpolate.BSpline(t, c, k)

        # for n+k+1 knots and degree k need at least n coefficients
        t = [0, 1, 2]
        c = [1]
        scp.interpolate.BSpline(t, c, k)

        t = [0, 1, 2, 3, 4]
        c = [1., 1.]
        k = 2
        scp.interpolate.BSpline(t, c, k)

        # non-integer orders
        t = [0., 0., 1., 2., 3., 4.]
        c = [1., 1., 1.]
        k = "cubic"
        scp.interpolate.BSpline(t, c, k)

        t = [0., 0., 1., 2., 3., 4.]
        c = [1., 1., 1.]
        k = 2.5
        scp.interpolate.BSpline(t, c, k)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        k = 2
        t = xp.arange(7, dtype=dtype)
        c = xp.asarray(self.c, dtype=dtype)
        test_xs = xp.linspace(-5, 10, 100, dtype=dtype)
        B = scp.interpolate.BSpline(t, c, k, extrapolate=self.extrapolate)
        return B(test_xs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_degree_1(self, xp, scp):
        t = xp.asarray([0, 1, 2, 3, 4])
        c = xp.asarray([1, 2, 3])
        k = 1

        b = scp.interpolate.BSpline(t, c, k)
        x = xp.linspace(1, 3, 50)
        return b(x)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_rndm_unity(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        b = self._make_random_spline(xp, scp)
        b.c = xp.ones_like(b.c)
        xx = xp.linspace(b.t[b.k], b.t[-b.k-1], 100)
        return b(xx)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vectorization(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        n, k = 22, 3
        t = xp.sort(xp.random.random(n))
        c = xp.random.random(size=(n, 6, 7))
        b = scp.interpolate.BSpline(t, c, k)
        tm, tp = t[k], t[-k-1]
        xx = tm + (tp - tm) * xp.random.random((3, 4, 5))
        return b(xx).shape

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-3)
    def test_bspline_len_c(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        # for n+k+1 knots, only first n coefs are used.
        n, k = 33, 3
        t = xp.sort(testing.shaped_random((n + k + 1,), xp))
        c = testing.shaped_random((n,), xp)

        # pad coefficients with random garbage
        c_pad = xp.r_[c, testing.shaped_random((k + 1,), xp)]

        BSpline = scp.interpolate.BSpline
        b, b_pad = BSpline(t, c, k), BSpline(t, c_pad, k)

        dt = t[-1] - t[0]

        # Locally, linspace produces relatively different values (1e-7) between
        # NumPy and CuPy during testing. Such difference result in a 1e-3
        # tolerance
        xx = xp.linspace(t[0] - dt, t[-1] + dt, 50, dtype=dtype)
        return b(xx), b_pad(xx)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_endpoints(self, xp, scp):
        # base interval is closed
        b = self._make_random_spline(xp, scp)
        t, _, k = b.tck
        tm, tp = t[k], t[-k-1]
        return (
            b(xp.asarray([tm, tp]), extrapolate=self.extrapolate),
            b(xp.asarray([tm + 1e-10, tp - 1e-10]),
              extrapolate=self.extrapolate))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_continuity(self, xp, scp):
        # assert continuity at internal knots
        b = self._make_random_spline(xp, scp)
        t, _, k = b.tck
        return b(t[k+1:-k-1] - 1e-10), b(t[k+1:-k-1] + 1e-10)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_extrap(self, xp, scp):
        b = self._make_random_spline(xp, scp)
        t, c, k = b.tck
        dt = t[-1] - t[0]
        xx = xp.linspace(t[k] - dt, t[-k-1] + dt, 50)
        mask = (t[k] < xx) & (xx < t[-k-1])

        return (b(xx[mask], extrapolate=self.extrapolate),
                b(xx, extrapolate=self.extrapolate))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_default_extrap(self, xp, scp):
        # BSpline defaults to extrapolate=True
        b = self._make_random_spline(xp, scp)
        t, _, k = b.tck
        xx = [t[0] - 1, t[-1] + 1]
        yy = b(xx)
        return not xp.all(xp.isnan(yy))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_periodic_extrap(self, xp, scp):
        b = self._make_random_spline(xp, scp, n=4, k=3)
        t, c, k = b.tck
        n = t.size - (k + 1)

        # Direct check
        xx = xp.asarray([-1, 0, 0.5, 1])
        xy = t[k] + (xx - t[k]) % (t[n] - t[k])
        return b(xx, extrapolate='periodic'), b(xy, extrapolate=True)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_derivative_rndm(self, xp, scp):
        b = self._make_random_spline(xp, scp)
        t, _, k = b.tck
        xx = xp.linspace(t[0], t[-1], 50)
        xx = xp.r_[xx, t]

        derivatives = []
        for der in range(1, k+2):
            derivatives.append(b(xx, nu=der))

        return derivatives

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bspline_derivative_jumps(self, xp, scp):
        # example from de Boor, Chap IX, example (24)
        # NB: knots augmented & corresp coefs are zeroed out
        # in agreement with the convention (29)
        k = 2
        t = xp.asarray([-1, -1, 0, 1, 1, 3, 4, 6, 6, 6, 7, 7])
        c = xp.r_[0, 0, testing.shaped_random((5,), xp), 0, 0]
        b = scp.interpolate.BSpline(t, c, k)

        comp = []
        # b is continuous at x != 6 (triple knot)
        x = xp.asarray([1, 3, 4, 6])
        comp.append(b(x[x != 6] - 1e-10))
        comp.append(b(x[x != 6] + 1e-10))
        comp.append(b(6. - 1e-10))
        comp.append(b(6. + 1e-10))

        # 1st derivative jumps at double knots, 1 & 6:
        x0 = xp.asarray([3, 4])
        comp.append(b(x0 - 1e-10, nu=1))
        comp.append(b(x0 + 1e-10, nu=1))

        x1 = xp.asarray([1, 6])
        comp.append(b(x1 - 1e-10, nu=1))
        comp.append(b(x1 + 1e-10, nu=1))

        # 2nd derivative is not guaranteed to be continuous either
        comp.append(b(x - 1e-10, nu=2))
        comp.append(b(x + 1e-10, nu=2))

        return comp

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basis_element(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        t = xp.arange(7, dtype=dtype)
        b = scp.interpolate.BSpline.basis_element(
            t, extrapolate=self.extrapolate)
        return b.tck

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basis_element_quadratic(self, xp, scp):
        xx = xp.linspace(-1, 4, 20)
        b = scp.interpolate.BSpline.basis_element(
            t=xp.asarray([0, 1, 2, 3]))
        r1 = b(xx)

        b = scp.interpolate.BSpline.basis_element(
            t=xp.asarray([0, 1, 1, 2]))
        xx = xp.linspace(0, 2, 10)
        r2 = b(xx)
        return r1, r2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basis_element_rndm(self, xp, scp):
        b = self._make_random_spline(xp, scp)
        t, c, k = b.tck
        xx = xp.linspace(t[k], t[-k-1], 20)
        n = len(t) - (k+1)
        s = 0.
        for i in range(n):
            b = scp.interpolate.BSpline.basis_element(
                t[i:i+k+2], extrapolate=False)(xx)
            # zero out out-of-bounds elements
            s += c[i] * xp.nan_to_num(b)
        return s

    @pytest.mark.xfail(
        runtime.is_hip and driver.get_build_version() < 5_00_00000,
        reason='name_expression with ROCm 4.3 may not work')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cmplx(self, xp, scp):
        b = self._make_random_spline(xp, scp)
        t, c, k = b.tck
        cc = c * (1. + 3.j)

        b = scp.interpolate.BSpline(t, cc, k)
        b_re = scp.interpolate.BSpline(t, b.c.real, k)
        b_im = scp.interpolate.BSpline(t, b.c.imag, k)

        xx = xp.linspace(t[k], t[-k-1], 20)
        return b(xx), b_re(xx) + 1j * b_im(xx)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nan(self, xp, scp):
        # nan in, nan out.
        b = scp.interpolate.BSpline.basis_element(
            xp.asarray([0, 1, 1, 2]))
        return b(xp.asarray([xp.nan]))

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @testing.with_requires('scipy>=1.8.0')
    def test_design_matrix(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        t = xp.arange(-1, 7, dtype=dtype)
        x = xp.arange(1, 5, dtype=dtype)
        k = 2

        mat = scp.interpolate.BSpline.design_matrix(x, t, k)
        return mat.todense()

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_single_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        k = 2
        t = xp.arange(7, dtype=dtype)
        c = xp.asarray(self.c, dtype=dtype)

        b = scp.interpolate.BSpline(t, c, k, extrapolate=self.extrapolate)
        return b.derivative().tck

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multiple_derivative(self, xp, scp):
        b = self._make_random_spline(xp, scp, k=5)
        t, c, k = b.tck
        xx = xp.linspace(t[k], t[-k-1], 20)
        comp = []
        for j in range(1, k):
            b = b.derivative()
            comp.append(b(xx))
        return comp

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antiderivative_tck(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        k = 2
        t = xp.arange(7, dtype=dtype)
        c = xp.asarray(self.c, dtype=dtype)

        b = scp.interpolate.BSpline(t, c, k, extrapolate=self.extrapolate)
        return b.antiderivative().tck

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antiderivative(self, xp, scp):
        b = self._make_random_spline(xp, scp)
        t, c, k = b.tck
        xx = xp.linspace(t[k], t[-k-1], 20)
        r1 = b.antiderivative().derivative()(xx)

        # repeat with N-D array for c
        c = xp.c_[c, c, c]
        c = xp.dstack((c, c))
        b = scp.interpolate.BSpline(t, c, k)
        r2 = b.antiderivative().derivative()(xx)
        return r1, r2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integral(self, xp, scp):
        # x for x < 1 else 2 - x
        b = scp.interpolate.BSpline.basis_element(xp.asarray([0, 1, 2]))

        ret = []
        ret.append(b.integrate(0, 1))
        ret.append(b.integrate(1, 0))
        ret.append(b.integrate(1, 0))

        # extrapolate or zeros outside of [0, 2]; default is yes
        ret.append(b.integrate(-1, 1))
        ret.append(b.integrate(-1, 1, extrapolate=True))
        ret.append(b.integrate(-1, 1, extrapolate=False))
        ret.append(b.integrate(1, -1, extrapolate=False))

        # Test ``_fitpack._splint()``
        ret.append(b.integrate(1, -1, extrapolate=False))

        # Test ``extrapolate='periodic'``.
        b.extrapolate = 'periodic'

        ret.append(b.integrate(0, 2))
        ret.append(b.integrate(2, 0))
        ret.append(b.integrate(-9, -7))
        ret.append(b.integrate(-8, -4))
        ret.append(b.integrate(0.5, 1.5))
        ret.append(b.integrate(1.5, 3))
        ret.append(b.integrate(1.5 + 12, 3 + 12))
        ret.append(b.integrate(1.5, 3 + 12))
        ret.append(b.integrate(0, -1))
        ret.append(b.integrate(-9, -10))
        ret.append(b.integrate(0, -9))
        return [xp.asarray(x) for x in ret]

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        k = 2
        t = xp.arange(7, dtype=dtype)
        c = xp.asarray(self.c, dtype=dtype)

        b = scp.interpolate.BSpline(t, c, k, extrapolate=self.extrapolate)
        return xp.asarray(b.integrate(0, 5))

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=True)
    def test_axis(self, xp, scp):
        n, k = 22, 3
        t = xp.linspace(0, 1, n + k + 1)
        x = testing.shaped_random((3, 4, 5), xp)

        ret = []
        for ax in range(-4, 4):
            sh = [6, 7, 8]
            # We need the positive axis for some of the indexing and
            # slices used in this test.
            pos_axis = self.axis % 4
            sh.insert(pos_axis, n)   # [22, 6, 7, 8] etc
            c = testing.shaped_random(sh, xp)
            b = scp.interpolate.BSpline(t, c, k, axis=ax)
            ret.append(b(x))

        # -c.ndim <= axis < c.ndim
        for ax in [-c.ndim - 1, c.ndim]:
            scp.interpolate.BSpline(t, c, k, axis=ax)

        # derivative, antiderivative keeps the axis
        BSpline = scp.interpolate.BSpline
        for b1 in [BSpline(t, c, k, axis=self.axis).derivative(),
                   BSpline(t, c, k, axis=self.axis).derivative(2),
                   BSpline(t, c, k, axis=self.axis).antiderivative(),
                   BSpline(t, c, k, axis=self.axis).antiderivative(2)]:
            ret.append(b1.axis)
        return b1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_neg_axis(self, xp, scp):
        k = 2
        t = xp.asarray([0, 1, 2, 3, 4, 5, 6])
        c = xp.asarray([[-1, 2, 0, -1], [2, 0, -3, 1]])

        spl = scp.interpolate.BSpline(t, c, k, axis=-1)
        return spl(xp.asarray([2.5]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @testing.with_requires('scipy>=1.8.0')
    def test_design_matrix_same_as_BSpline_call(self, xp, scp):
        """Test that design_matrix(x) is equivalent to BSpline(..)(x)."""
        ret = []

        has_extrapolate = True
        for mod in [cupyx.scipy.interpolate, interpolate]:
            sig = inspect.signature(mod.BSpline.design_matrix)
            has_extrapolate = has_extrapolate and (
                'extrapolate' in sig.parameters)

        kwargs = {}
        if has_extrapolate:
            kwargs = {'extrapolate': self.extrapolate}

        for k in range(0, 5):
            np.random.seed(1234)
            x = xp.asarray(np.random.random_sample(10 * (k + 1)))
            xmin, xmax = xp.amin(x), xp.amax(x)

            t = xp.r_[xp.linspace(xmin - 2, xmin - 1, k),
                      xp.linspace(xmin, xmax, 2 * (k + 1)),
                      xp.linspace(xmax + 1, xmax + 2, k)]
            c = xp.eye(len(t) - k - 1)
            bspline = scp.interpolate.BSpline(t, c, k, self.extrapolate)
            ret.append(bspline(x))
            ret.append(scp.interpolate.BSpline.design_matrix(
                x, t, k, **kwargs).todense())

            # extrapolation regime
            if has_extrapolate:
                x = xp.array([xmin - 10, xmin - 1, xmax + 1.5, xmax + 10])
                if not self.extrapolate:
                    scp.interpolate.BSpline.design_matrix(
                        x, t, k, self.extrapolate)
                else:
                    ret.append(bspline(x))
                    ret.append(scp.interpolate.BSpline.design_matrix(
                        x, t, k, self.extrapolate).todense())

        return ret

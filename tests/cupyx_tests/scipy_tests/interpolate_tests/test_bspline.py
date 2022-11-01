import pytest

from cupy import testing
import cupyx.scipy.interpolate  # NOQA

try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'extrapolate': [True, False],
    'c': [[-1, 2, 0, -1], [[-1, 2, 0, -1]] * 5]}))
@testing.with_requires("scipy")
class TestBSpline:

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

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basis_element(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        t = xp.arange(7, dtype=dtype)
        b = scp.interpolate.BSpline.basis_element(
            t, extrapolate=self.extrapolate)
        return b.tck

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
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
    def test_derivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        k = 2
        t = xp.arange(7, dtype=dtype)
        c = xp.asarray(self.c, dtype=dtype)

        b = scp.interpolate.BSpline(t, c, k, extrapolate=self.extrapolate)
        return b.derivative().tck

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antiderivative(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()

        k = 2
        t = xp.arange(7, dtype=dtype)
        c = xp.asarray(self.c, dtype=dtype)

        b = scp.interpolate.BSpline(t, c, k, extrapolate=self.extrapolate)
        return b.antiderivative().tck

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

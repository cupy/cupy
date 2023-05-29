import numpy

from cupy import testing
import cupyx.scipy.special  # NOQA


@testing.with_requires('scipy')
class TestBinom:
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA
        n = testing.shaped_arange((40, 100), xp, dtype) + 20
        k = testing.shaped_arange((40, 100), xp, dtype)
        return scp.special.binom(n, k)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype):
        import scipy.special  # NOQA
        n = xp.linspace(30, 60, 1000, dtype=dtype)
        k = xp.linspace(15, 60, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace_largen(self, xp, scp, dtype):
        import scipy.special  # NOQA
        n = xp.linspace(1e10, 9e10, 1000, dtype=dtype)
        k = xp.linspace(.01, .9, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace_largeposk(self, xp, scp, dtype):
        import scipy.special  # NOQA
        n = xp.linspace(.01, .9, 1000, dtype=dtype)
        k = xp.linspace(1e10+.5, 9e10+.5, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace_largenegk(self, xp, scp, dtype):
        import scipy.special  # NOQA
        n = xp.linspace(.01, .9, 1000, dtype=dtype)
        k = xp.linspace(.5-1e10, .5-9e10, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_nan_inf(self, xp, scp, dtype):
        import scipy.special  # NOQA
        a = xp.array([-numpy.inf, numpy.nan, numpy.inf, 0, -1,
                      1e8, 5e7], dtype=dtype)
        return scp.special.binom(a[:, xp.newaxis], a[xp.newaxis, :])

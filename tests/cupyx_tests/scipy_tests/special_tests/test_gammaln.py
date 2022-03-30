import math

import numpy
import pytest

from cupy import testing
import cupyx.scipy.special  # NOQA


@testing.gpu
@testing.with_requires('scipy')
class TestGammaln:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        return scp.special.gammaln(a)

    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-5, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(-30, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        return scp.special.gammaln(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        return scp.special.gammaln(dtype(1.5))

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        return scp.special.gammaln(a)


@testing.gpu
@testing.with_requires('scipy')
class TestMultigammaln:

    @pytest.mark.parametrize('d', [1, 5, 15])
    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-5, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, d):
        import scipy.special  # NOQA

        minval = math.ceil(0.5 * (d - 1) + .0001)
        a = xp.linspace(minval, minval + 50, 1000, dtype=dtype)
        return scp.special.multigammaln(a, d)

    @pytest.mark.parametrize('d', [1, 5, 15])
    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype, d):
        import scipy.special  # NOQA

        return scp.special.multigammaln(dtype(30), d)

    @pytest.mark.parametrize('d', [1, 5, 15])
    @pytest.mark.parametrize('a', ['nan', 'inf'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_nonfinite_scalar(self, xp, scp, dtype, a, d):
        import scipy.special  # NOQA

        a = getattr(xp, a)
        return scp.special.multigammaln(dtype(a), d)

    @pytest.mark.parametrize('d', [1, 5, 15])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_nonfinite_array(self, xp, scp, dtype, d):
        import scipy.special  # NOQA

        a = xp.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)

        # Note: TypeError is only here because of a bug in SciPy's formatting
        # of the ValueError message
        with pytest.raises((ValueError, TypeError)):
            scp.special.multigammaln(a, d)
        return xp.zeros(0)

    @pytest.mark.parametrize('d', ['array', 1.5])
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_invalid_d(self, xp, scp, dtype, d):
        import scipy.special  # NOQA

        a = xp.array([5.0, 10.0, 15.0]).astype(dtype)
        if d == 'array':
            d = xp.arange(10, 20, dtype=int)

        with pytest.raises(ValueError):
            scp.special.multigammaln(a, d)
        return xp.zeros(0)

    @pytest.mark.parametrize('d', [1, 5, 15])
    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-5, scipy_name='scp')
    def test_invalid_a(self, xp, scp, dtype, d):
        import scipy.special  # NOQA

        # all values must be > (0.5 * (d - 1)) so this minval is invalid
        minval = 0.5 * (d - 1)
        a = xp.linspace(minval, minval + 10, 10, dtype=dtype)

        # Note: TypeError is only here because of a bug in SciPy's formatting
        # of the ValueError message
        with pytest.raises((ValueError, TypeError)):
            scp.special.multigammaln(a, d)
        return xp.zeros(0)

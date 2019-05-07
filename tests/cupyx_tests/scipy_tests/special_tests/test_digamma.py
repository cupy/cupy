import unittest

from cupy import testing
import cupyx.scipy.special  # NOQA
import numpy


@testing.gpu
@testing.with_requires('scipy')
class TestDigamma(unittest.TestCase):

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        return scp.special.digamma(a)

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_linspace_positive(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(0, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        return scp.special.digamma(a)

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_linspace_negative(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(-30, 0, 1000, dtype=dtype)
        a = xp.asarray(a)
        return scp.special.digamma(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        return scp.special.digamma(dtype(1.5))

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        return scp.special.digamma(a)

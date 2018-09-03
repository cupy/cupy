import unittest

from cupy import testing
import cupyx.scipy.special  # NOQA
import numpy


@testing.gpu
@testing.with_requires('scipy')
class TestGammaln(unittest.TestCase):

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

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        return scp.special.gammaln(a)

import unittest

import cupy
from cupy import testing
import cupyx.scipy.special
import numpy

try:
    import scipy.special
    _scipy_available = True
except ImportError:
    _scipy_available = False


@testing.gpu
@testing.with_requires('scipy')
class TestGammaln(unittest.TestCase):

    def _get_xp_func(self, xp):
        if xp is cupy:
            return cupyx.scipy.special
        else:
            return scipy.special

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_arange(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return self._get_xp_func(xp).gammaln(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-5)
    def test_linspace(self, xp, dtype):
        a = numpy.linspace(-30, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        return self._get_xp_func(xp).gammaln(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3)
    def test_scalar(self, xp, dtype):
        return self._get_xp_func(xp).gammaln(dtype(1.5))

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3)
    def test_inf_and_nan(self, xp, dtype):
        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        return self._get_xp_func(xp).gammaln(a)

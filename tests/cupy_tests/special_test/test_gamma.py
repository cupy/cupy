import unittest

from cupy import testing
import numpy


@testing.gpu
@testing.with_requires('scipy')
class TestGamma(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, mod='sp', mod_name='special')
    def test_arange(self, xp, dtype, sp):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return sp.gamma(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, mod='sp',
                                 mod_name='special')
    def test_linspace(self, xp, dtype, sp):
        a = numpy.linspace(-30, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        return sp.gamma(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, mod='sp',
                                 mod_name='special')
    def test_scalar(self, xp, dtype, sp):
        return sp.gamma(dtype(1.5))

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, mod='sp',
                                 mod_name='special')
    def test_inf_and_nan(self, xp, dtype, sp):
        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        return sp.gamma(a)

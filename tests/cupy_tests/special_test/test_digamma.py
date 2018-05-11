import unittest

from cupy import testing
import numpy


@testing.gpu
@testing.with_requires('scipy')
class TestDigamma(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, mod='sp', mod_name='special')
    def test_arange(self, xp, dtype, sp):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return sp.digamma(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6, mod='sp',
                                 mod_name='special')
    def test_linspace_positive(self, xp, dtype, sp):
        if (dtype == xp.dtype('B') or dtype == xp.dtype('H')
            or dtype == xp.dtype('I') or dtype == xp.dtype('L')
                or dtype == xp.dtype('Q')):
            a = numpy.linspace(0, 30, 1000, dtype=dtype)
            a = xp.asarray(a)
        else:
            a = xp.linspace(0, 30, 1000, dtype=dtype)
        return sp.digamma(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, mod='sp',
                                 mod_name='special')
    def test_linspace_negative(self, xp, dtype, sp):
        if (dtype == xp.dtype('B') or dtype == xp.dtype('H')
            or dtype == xp.dtype('I') or dtype == xp.dtype('L')
                or dtype == xp.dtype('Q')):
            a = numpy.linspace(-30, 0, 1000, dtype=dtype)
            a = xp.asarray(a)
        else:
            a = xp.linspace(-30, 0, 1000, dtype=dtype)
        return sp.digamma(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, mod='sp',
                                 mod_name='special')
    def test_scalar(self, xp, dtype, sp):
        return sp.digamma(dtype(1.5))

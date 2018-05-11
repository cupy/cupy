import unittest

from cupy import testing
import numpy


@testing.gpu
@testing.with_requires('scipy')
class TestZeta(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, mod='sp', mod_name='special')
    def test_arange(self, xp, dtype, sp):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_arange((2, 3), xp, dtype)
        return sp.zeta(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6, mod='sp',
                                 mod_name='special')
    def test_linspace(self, xp, dtype, sp):
        if (dtype == xp.dtype('B') or dtype == xp.dtype('H')
            or dtype == xp.dtype('I') or dtype == xp.dtype('L')
                or dtype == xp.dtype('Q')):
            a = numpy.linspace(-30, 30, 1000, dtype=dtype)
            b = numpy.linspace(-30, 30, 1000, dtype=dtype)
            a = xp.asarray(a)
            b = xp.asarray(b)
        else:
            a = xp.linspace(-30, 30, 1000, dtype=dtype)
            b = xp.linspace(-30, 30, 1000, dtype=dtype)
        return sp.zeta(a, b)

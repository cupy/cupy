import unittest

import cupy
from cupy import testing
import numpy
import cupyx.scipy.special

import scipy.special


@testing.gpu
@testing.with_requires('scipy')
class TestZeta(unittest.TestCase):

    def _get_xp_func(self, xp):
        if xp is cupy:
            return cupyx.scipy.special
        else:
            return scipy.special

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_arange(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_arange((2, 3), xp, dtype)
        return self._get_xp_func(xp).zeta(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6)
    def test_linspace(self, xp, dtype):
        a = numpy.linspace(-30, 30, 1000, dtype=dtype)
        b = numpy.linspace(-30, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        b = xp.asarray(b)
        return self._get_xp_func(xp).zeta(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3)
    def test_scalar(self, xp, dtype):
        return self._get_xp_func(xp).zeta(dtype(2.), dtype(1.5))

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3)
    def test_inf_and_nan(self, xp, dtype):
        x = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = numpy.tile(x, 3)
        b = numpy.repeat(x, 3)
        a = xp.asarray(a)
        b = xp.asarray(b)
        return self._get_xp_func(xp).zeta(a, b)

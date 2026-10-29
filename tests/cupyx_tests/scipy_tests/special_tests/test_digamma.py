from __future__ import annotations

import unittest

from cupy import testing
import cupyx.scipy.special  # NOQA
import numpy
from cupyx_tests.scipy_tests.special_tests import match_scipy_float32


@testing.with_requires("scipy")
class TestDigamma(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-15, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        out = scp.special.digamma(a)
        return match_scipy_float32(out, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_linspace_positive(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(0, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        out = scp.special.digamma(a)
        return match_scipy_float32(out, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_linspace_negative(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(-30, 0, 1000, dtype=dtype)
        a = xp.asarray(a)
        out = scp.special.digamma(a)
        return match_scipy_float32(out, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        out = scp.special.digamma(dtype(1.5))
        return match_scipy_float32(out, xp, dtype)

    @testing.with_requires('scipy')
    @testing.for_dtypes("efdFD")
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        out = scp.special.digamma(a)
        return match_scipy_float32(out, xp, dtype)

    def test_psi(self):
        """Verify that psi exists and is the same as digamma"""
        assert cupyx.scipy.special.psi is cupyx.scipy.special.digamma

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_complex(self, xp, scp, dtype):
        x = xp.linspace(-20, 20, 12, dtype=dtype)
        y = xp.linspace(-20, 20, 12, dtype=dtype)
        x, y = xp.meshgrid(x, y)
        z = (x + y*1j).ravel()
        return scp.special.digamma(z)

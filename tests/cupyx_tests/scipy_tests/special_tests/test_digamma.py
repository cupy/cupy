from __future__ import annotations

import unittest

from cupy import testing
import cupyx.scipy.special  # NOQA
import numpy


# NOTE: float16 is left out of the dtype sweeps below.  SciPy 1.18 resolves
# float16 input to the float32 loop of the `scipy.special` ufuncs and returns
# float32, whereas CuPy's ufuncs declare `e->d` and return float64.
# TODO: switch those loops to `e->f` and restore float16 coverage before the
# minimum SciPy version is 1.18.
@testing.with_requires("scipy")
class TestDigamma(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-15, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        return scp.special.digamma(a)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_linspace_positive(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(0, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        return scp.special.digamma(a)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_linspace_negative(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.linspace(-30, 0, 1000, dtype=dtype)
        a = xp.asarray(a)
        return scp.special.digamma(a)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        return scp.special.digamma(dtype(1.5))

    @testing.with_requires('scipy')
    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        return scp.special.digamma(a)

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

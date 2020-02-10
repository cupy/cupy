import unittest

from cupy import testing
import cupyx.scipy.special  # NOQA
import numpy

import warnings


@testing.gpu
@testing.with_requires('scipy')
class TestPolygamma(unittest.TestCase):

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_arange((2, 3), xp, dtype)
        return scp.special.polygamma(a, b)

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = numpy.tile(numpy.arange(5), 200).astype(dtype)
        b = numpy.linspace(-30, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        b = xp.asarray(b)
        return scp.special.polygamma(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        # polygamma in scipy returns numpy.float64 value when inputs scalar.
        # whatever type input is.
        return scp.special.polygamma(
            dtype(2.), dtype(1.5)).astype(numpy.float32)

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        x = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = numpy.tile(x, 3)
        b = numpy.repeat(x, 3)
        a = xp.asarray(a)
        b = xp.asarray(b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return scp.special.polygamma(a, b)

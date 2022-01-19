import unittest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.special


class _TestBase(object):

    def test_ndtr(self):
        self.check_unary('ndtr')

    def test_expit(self):
        self.check_unary_lower_precision('expit')

    @testing.with_requires('scipy>=1.8.0rc0')
    def test_log_expit(self):
        self.check_unary_lower_precision('log_expit')


atol = {'default': 1e-15, cupy.float64: 1e-15}
rtol = {'default': 1e-5, cupy.float64: 1e-15}

# not all functions pass at the stricter tolerances above
atol_low = {'default': 5e-4, cupy.float64: 1e-12}
rtol_low = {'default': 5e-4, cupy.float64: 1e-12}


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial(unittest.TestCase, _TestBase):

    def _check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        return self._check_unary(name, xp, scp, dtype)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def check_unary_lower_precision(self, name, xp, scp, dtype):
        return self._check_unary(name, xp, scp, dtype)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def test_logit(self, xp, scp, dtype):
        import scipy.special  # NOQA

        # outputs are only finite over range (0, 1)
        a = xp.linspace(0.001, .999, 1000, dtype=dtype)
        return scp.special.logit(a)

    def test_logit_nonfinite(self):
        assert float(cupyx.scipy.special.logit(0)) == -numpy.inf
        assert float(cupyx.scipy.special.logit(1)) == numpy.inf
        assert numpy.isnan(float(cupyx.scipy.special.logit(1.1)))
        assert numpy.isnan(float(cupyx.scipy.special.logit(-0.1)))


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase, _TestBase):

    def _check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        return self._check_unary(name, xp, scp, dtype)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def check_unary_lower_precision(self, name, xp, scp, dtype):
        return self._check_unary(name, xp, scp, dtype)

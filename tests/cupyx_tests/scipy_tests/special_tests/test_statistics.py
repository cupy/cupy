import unittest

import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.special


class _TestBase:

    def test_ndtr(self):
        self.check_unary_linspace0_1('ndtr')

    def test_ndtri(self):
        self.check_unary_linspace0_1('ndtri')

    def test_logit(self):
        self.check_unary_lower_precision('logit')

    def test_expit(self):
        self.check_unary_lower_precision('expit')

    @testing.with_requires('scipy>=1.8.0rc0')
    def test_log_expit(self):
        self.check_unary_lower_precision('log_expit')


atol = {'default': 1e-14, cupy.float64: 1e-14}
rtol = {'default': 1e-5, cupy.float64: 1e-14}

# not all functions pass at the stricter tolerances above
atol_low = {'default': 5e-4, cupy.float64: 1e-12}
rtol_low = {'default': 5e-4, cupy.float64: 1e-12}


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial(_TestBase):

    def _check_unary(self, a, name, scp):
        import scipy.special  # NOQA

        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def check_unary_lower_precision(self, name, xp, scp, dtype):
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary_linspace0_1(self, name, xp, scp, dtype):
        p = xp.linspace(0, 1, 1000, dtype=dtype)
        return self._check_unary(p, name, scp)

    def test_logit_nonfinite(self):
        logit = cupyx.scipy.special.logit
        assert float(logit(0)) == -numpy.inf
        assert float(logit(1)) == numpy.inf
        assert numpy.isnan(float(logit(1.1)))
        assert numpy.isnan(float(logit(-0.1)))

    @pytest.mark.parametrize('inverse', [False, True],
                             ids=['boxcox', 'inv_boxcox'])
    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def test_boxcox(self, xp, scp, dtype, inverse):
        import scipy.special  # NOQA

        # outputs are only finite over range (0, 1)
        x = xp.linspace(0.001, 1000, 1000, dtype=dtype).reshape((1, 1000))
        lmbda = xp.asarray([-5, 0, 5], dtype=dtype).reshape((3, 1))
        result = scp.special.boxcox(x, lmbda)
        if inverse:
            result = scp.special.inv_boxcox(result, lmbda)
        return result

    def test_boxcox_nonfinite(self):
        boxcox = cupyx.scipy.special.boxcox
        assert float(boxcox(0, -5)) == -numpy.inf
        assert numpy.isnan(float(boxcox(-0.1, 5)))

    @pytest.mark.parametrize('inverse', [False, True],
                             ids=['boxcox', 'inv_boxcox'])
    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def test_boxcox1p(self, xp, scp, dtype, inverse):
        import scipy.special  # NOQA

        x = xp.linspace(-0.99, 1000, 1000, dtype=dtype).reshape((1, 1000))
        lmbda = xp.asarray([-5, 0, 5], dtype=dtype).reshape((3, 1))
        result = scp.special.boxcox1p(x, lmbda)
        if inverse:
            result = scp.special.inv_boxcox1p(result, lmbda)
        return result

    def test_boxcox1p_nonfinite(self):
        boxcox1p = cupyx.scipy.special.boxcox1p
        assert float(boxcox1p(-1, -5)) == -numpy.inf
        assert numpy.isnan(float(boxcox1p(-1.1, 5)))


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase, _TestBase):

    def _check_unary(self, a, name, scp):
        import scipy.special  # NOQA

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def check_unary_lower_precision(self, name, xp, scp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary_linspace0_1(self, name, xp, scp, dtype):
        a = xp.linspace(0, 1, 1000, dtype)
        return self._check_unary(a, name, scp)

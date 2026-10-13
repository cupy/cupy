from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing._helper import installed_but_not_baseline
import cupyx.scipy.special  # NOQA
from cupyx_tests.scipy_tests.special_tests import match_scipy_float32


@testing.with_requires("scipy>=1.15")
class TestGamma:

    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        func = getattr(scp.special, function)
        out = func(a)
        return match_scipy_float32(out, xp, dtype)

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip and
        cupy.cuda.runtime.runtimeGetVersion() < 5_00_00000,
        reason='ROCm/HIP fails in ROCm 4.x')
    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        a = numpy.linspace(-30, 30, 1000, dtype=dtype)
        if a.dtype.kind == 'c':
            a -= 1j * a
        a = xp.asarray(a)
        func = getattr(scp.special, function)
        out = func(a)
        return match_scipy_float32(out, xp, dtype)

    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        if xp.dtype(dtype).kind == 'c':
            val = dtype(1.5 + 1.0j)
        else:
            val = dtype(1.5)
        func = getattr(scp.special, function)
        out = func(val)
        return match_scipy_float32(out, xp, dtype)

    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_dtypes("efdFD")
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    @testing.with_requires('scipy')
    def test_inf_and_nan(self, xp, scp, dtype, function):
        import scipy.special  # NOQA
        if (installed_but_not_baseline(scipy="1.18") and function == 'rgamma'
                and xp.dtype(dtype).kind == 'c'):
            pytest.skip("SciPy 1.18 returns nan for complex rgamma(-inf) "
                        "where CuPy still returns 0.")

        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        func = getattr(scp.special, function)
        out = func(a)
        return match_scipy_float32(out, xp, dtype)

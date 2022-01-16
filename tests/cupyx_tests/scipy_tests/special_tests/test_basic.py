import math

import cupy
import numpy
import pytest
import scipy.special  # NOQA

import cupyx.scipy.special  # NOQA
from cupy import testing
from cupy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
)
from cupy.testing import numpy_cupy_allclose


@testing.gpu
@testing.with_requires("scipy")
class TestLegendreFunctions:

    def test_lpmv_basic(self):
        # specific values tested in the SciPy test suite
        scp = cupyx.scipy
        lp = scp.special.lpmv(0, 2, 0.5)
        assert_array_almost_equal(lp, -0.125, 7)
        lp = scp.special.lpmv(0, 40, 0.001)
        assert_array_almost_equal(lp, 0.1252678976534484, 7)

        # XXX: this is outside the domain of the current implementation,
        #      so ensure it returns a NaN rather than a wrong answer.
        olderr = numpy.seterr(all="ignore")
        try:
            lp = scp.special.lpmv(-1, -1, 0.001)
        finally:
            numpy.seterr(**olderr)
        assert lp != 0 or cupy.isnan(lp)

    @pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", atol=1e-12)
    def test_lpmv(self, xp, scp, dtype, order, degree):
        vals = xp.linspace(-1, 1, 100, dtype=dtype)
        return scp.special.lpmv(order, degree, vals)


@testing.gpu
@testing.with_requires("scipy")
class TestBasic:

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_gammasgn(self, xp, scp, dtype):
        vals = xp.linspace(-4, 4, 100, dtype=dtype)
        return scp.special.gammasgn(vals)

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-5)
    def test_log1p_linspace(self, xp, scp, dtype):
        vals = xp.linspace(-100, 100, 1000, dtype=dtype)
        dtype = xp.dtype(dtype)
        if dtype.kind == 'c':
            # broadcast to mix large and small real vs. imaginary
            vals = vals[::10, xp.newaxis] + 1j * vals[xp.newaxis, ::10]
        return xp.abs(scp.special.log1p(vals))

    @testing.for_dtypes("efFdD")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-5)
    def test_log1p_logspace(self, xp, scp, dtype):
        dtype = xp.dtype(dtype)
        vals = xp.logspace(-10, 10, 1000, dtype=dtype)
        if dtype.kind == 'c':
            # broadcast to mix large and small real vs. imaginary
            vals = vals[::10, xp.newaxis] + 1j * vals[xp.newaxis, ::10]
        return xp.abs(scp.special.log1p(vals))

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_log1p_path2(self, xp, scp, dtype):
        # test values for code path corresponding to range [1/sqrt(2), sqrt(2)]
        vals = xp.linspace(1 / math.sqrt(2), math.sqrt(2), 1000, dtype=dtype)
        return scp.special.log1p(vals)

    def test_log1p_real(self):
        log1p = cupyx.scipy.special.log1p
        inf = cupy.inf
        nan = cupy.nan
        assert_array_equal(log1p(0), 0.0)
        assert_array_equal(log1p(-1), -inf)
        assert_array_equal(log1p(-2), nan)
        assert_array_equal(log1p(inf), inf)

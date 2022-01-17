import math

import cupy
import numpy
import pytest
import scipy.special  # NOQA

import cupyx.scipy.special
from cupy import testing
from cupy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
)
from cupy.testing import numpy_cupy_allclose

rtol = {'default': 1e-5, cupy.float64: 1e-12}


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
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", atol=1e-12)
    def test_lpmv(self, xp, scp, dtype, order, degree):
        vals = xp.linspace(-1, 1, 100, dtype=dtype)
        return scp.special.lpmv(order, degree, vals)


@testing.gpu
@testing.with_requires("scipy")
class TestBasic:
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp")
    def test_gammasgn(self, xp, scp, dtype):
        vals = xp.linspace(-4, 4, 100, dtype=dtype)
        return scp.special.gammasgn(vals)

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", rtol=rtol)
    def test_log1p_(self, xp, scp, dtype):
        # only test with values > 0 to avoid NaNs
        vals = xp.logspace(-10, 10, 10000, dtype=dtype)
        return scp.special.log1p(vals)

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", rtol=rtol)
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

    def test_log1p_complex(self):
        # complex-valued log1p not yet implemented
        with pytest.raises(TypeError):
            cupyx.scipy.special.log1p(0 + 0j)

    @pytest.mark.parametrize("function", ["xlogy", "xlog1py"])
    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name="scp", rtol={'default': 1e-3,
                                                 cupy.float64: 1e-12})
    def test_xlogy(self, xp, scp, dtype, function):
        # only test with values > 0 to avoid NaNs
        x = xp.linspace(-100, 100, 1000, dtype=dtype)
        y = xp.linspace(0.001, 100, 1000, dtype=dtype)
        if x.dtype.kind == 'c':
            x -= 1j * x
            y += 1j * y
        return getattr(scp.special, function)(x, y)

    @pytest.mark.parametrize("function", ["xlogy", "xlog1py"])
    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name="scp", rtol={'default': 1e-3,
                                                 cupy.float64: 1e-12})
    def test_xlogy_zeros(self, xp, scp, dtype, function):
        # only test with values > 0 to avoid NaNs
        x = xp.zeros((1, 100), dtype=dtype)
        y = xp.linspace(-10, 10, 100, dtype=dtype)
        if y.dtype.kind == 'c':
            y += 1j * y
        return getattr(scp.special, function)(x, y)

    @pytest.mark.parametrize("function", ["xlogy", "xlog1py"])
    @testing.for_all_dtypes()
    def test_xlogy_nonfinite(self, dtype, function):
        func = getattr(cupyx.scipy.special, function)
        y = cupy.ones((5,), dtype=dtype)
        assert cupy.all(cupy.isnan(func(cupy.nan, y)))
        assert cupy.all(cupy.isnan(func(y, cupy.nan)))

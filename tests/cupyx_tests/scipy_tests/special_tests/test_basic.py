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

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_log1p_(self, xp, scp, dtype):
        # only test with values > 0 to avoid NaNs
        vals = xp.logspace(-10, 10, 10000, dtype=dtype)
        return scp.special.log1p(vals)

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

    def test_log1p_complex(self):
        # complex-valued log1p not yet implemented
        with pytest.raises(TypeError):
            cupyx.scipy.special.log1p(0 + 0j)

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_exp2(self, xp, scp, dtype):
        vals = xp.linspace(-100, 100, 200, dtype=dtype)
        return scp.special.exp2(vals)

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_exp10(self, xp, scp, dtype):
        if xp.dtype(dtype).char == 'd':
            vals = xp.linspace(-100, 100, 100, dtype=dtype)
        else:
            # Note: comparisons start to fail outside this range
            #       np.finfo(np.float32).max is 3.4028235e+38
            vals = xp.linspace(-37, 37, 100, dtype=dtype)
        return scp.special.exp10(vals)

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_expm1(self, xp, scp, dtype):
        vals = xp.linspace(-100, 100, 200, dtype=dtype)
        return scp.special.expm1(vals)

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_radian(self, xp, scp, dtype):
        tmp = xp.linspace(-100, 100, 10, dtype=dtype)
        d = tmp[:, xp.newaxis, xp.newaxis]
        m = tmp[xp.newaxis, : xp.newaxis]
        s = tmp[xp.newaxis, xp.newaxis, :]
        return scp.special.radian(d, m, s)

    @pytest.mark.parametrize('function', ['cosdg', 'sindg', 'tandg', 'cotdg'])
    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", atol=1e-10, rtol=1e-6)
    def test_trig_degrees(self, xp, scp, dtype, function):
        vals = xp.linspace(-100, 100, 200, dtype=dtype)
        # test at exact multiples of 45 degrees
        vals = xp.concatenate((vals, xp.arange(-360, 361, 45, dtype=dtype)))
        return getattr(scp.special, function)(vals)

    @testing.for_dtypes("efd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_cbrt(self, xp, scp, dtype):
        vals = xp.linspace(-100, 100, 200, dtype=dtype)
        return scp.special.cbrt(vals)

    # Exclude 'e' here because of deficiency in the NumPy/SciPy
    # implementation for float16 dtype. This was also noted in
    # cupy_tests/math_tests/test_special.py
    @testing.for_dtypes("fd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6, atol=1e-6)
    def test_sinc(self, xp, scp, dtype):
        vals = xp.linspace(-100, 100, 200, dtype=dtype)
        return scp.special.sinc(vals)

    # TODO: Should we make all int dtypes convert to float64 and test for that?
    #       currently int8->float16, int16->float32, etc... but for
    #       numpy.sinc/scipy.special.sinc any int type becomes float64.
    @testing.for_dtypes("fd")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6, atol=1e-6)
    def test_sinc_integer_valued(self, xp, scp, dtype):
        vals = xp.arange(0, 10, dtype=dtype)
        return scp.special.sinc(vals)

import math
import unittest

import cupy
import numpy
import pytest
import scipy.special  # NOQA
import scipy.special._ufuncs as cephes
from numpy.testing import suppress_warnings

import cupyx.scipy.special  # NOQA
from cupy import testing
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
)
from cupy.testing import numpy_cupy_allclose


# TODO: update/expand lpmv tests. The below is adapted from SciPy
@testing.gpu
@testing.with_requires("scipy")
class TestLegendreFunctions(unittest.TestCase):
    def test_lpmv_basic(self):
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


def test_gammasgn_vs_cephes():
    vals = cupy.asarray([-4, -3.5, -2.3, 1, 4.2], cupy.float64)
    scp = cupyx.scipy
    assert_array_equal(
        scp.special.gammasgn(vals),
        numpy.sign(cephes.rgamma(cupy.asnumpy(vals))),
    )


@testing.gpu
@testing.with_requires("scipy")
class TestBasic(unittest.TestCase):
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp")
    def test_gammasgn(self, xp, scp, dtype):
        vals = xp.asarray([-4, -3.5, -2.3, 1, 4.2], dtype=dtype)
        return scp.special.gammasgn(vals)

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_log1p_(self, xp, scp, dtype):
        # only test with values > 0 to avoid NaNs
        vals = xp.logspace(-10, 10, 10000, dtype=dtype)
        return scp.special.log1p(vals)

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-6)
    def test_log1p_path2(self, xp, scp, dtype):
        # test values for code path corresponding to range [1/sqrt(2), sqrt(2)]
        vals = xp.linspace(1 / math.sqrt(2), math.sqrt(2), 1000, dtype=dtype)
        return scp.special.log1p(vals)


def test_log1p_real():
    log1p = cupyx.scipy.special.log1p
    inf = cupy.inf
    nan = cupy.nan
    assert_array_equal(log1p(0), 0.0)
    assert_array_equal(log1p(-1), -inf)
    assert_array_equal(log1p(-2), nan)
    assert_array_equal(log1p(inf), inf)


@pytest.mark.skip(reason="complex not currently supported in log1p")
def test_log1p_complex():
    log1p = cupyx.scipy.special.log1p
    c = complex
    inf, nan, pi = cupy.inf, cupy.nan, cupy.pi
    assert_array_equal(log1p(0 + 0j), 0 + 0j)
    assert_array_equal(log1p(c(-1, 0)), c(-inf, 0))
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "invalid value encountered in multiply")
        assert_allclose(log1p(c(1, inf)), c(inf, pi / 2))
        assert_array_equal(log1p(c(1, nan)), c(nan, nan))
        assert_allclose(log1p(c(-inf, 1)), c(inf, pi))
        assert_array_equal(log1p(c(inf, 1)), c(inf, 0))
        assert_allclose(log1p(c(-inf, inf)), c(inf, 3 * pi / 4))
        assert_allclose(log1p(c(inf, inf)), c(inf, pi / 4))
        assert_array_equal(log1p(c(inf, nan)), c(inf, nan))
        assert_array_equal(log1p(c(-inf, nan)), c(inf, nan))
        assert_array_equal(log1p(c(nan, inf)), c(inf, nan))
        assert_array_equal(log1p(c(nan, 1)), c(nan, nan))
        assert_array_equal(log1p(c(nan, nan)), c(nan, nan))

from __future__ import annotations

import sys

import numpy
import pytest

import cupy
from cupy import testing
import cupyx
import cupyx.scipy.stats  # NOQA


atol = {
    cupy.float16: 5e-3,
    cupy.float32: 1e-6,
    cupy.complex64: 1e-6,
    cupy.float64: 1e-14,
    cupy.complex128: 1e-14,
}
rtol = {
    cupy.float16: 0.1,
    cupy.float32: 1e-6,
    cupy.complex64: 1e-6,
    cupy.float64: 1e-14,
    cupy.complex128: 1e-14,
}


def _dtype(dtype, xp):
    dtype = xp.dtype(dtype)
    if dtype.kind in 'fc':
        return dtype
    if dtype in (xp.int8, xp.uint8):
        return xp.float16
    if dtype in (xp.int16, xp.uint16):
        return xp.float32
    return xp.float64


def _make_data(shape, xp, dtype):
    if dtype == xp.float16:
        return testing.shaped_random(shape, xp, dtype=dtype, scale=3)
    else:
        return testing.shaped_arange(shape, xp, dtype=dtype)


@testing.with_requires('scipy>=1.15')
class TestBoxcox_llf:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_1dim(self, xp, scp, dtype):
        data = _make_data((10,), xp, dtype)
        lmb = 4.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_2dim(self, xp, scp, dtype):
        data = _make_data((3, 8), xp, dtype)
        lmb = 6.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_3dim(self, xp, scp, dtype):
        data = _make_data((10, 3, 4), xp, dtype)
        lmb = 3.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_multi_dim(self, xp, scp, dtype):
        if dtype == xp.float16:
            data = _make_data((3, 2, 3, 2), xp, dtype)
        else:
            data = _make_data((3, 2, 4, 3), xp, dtype)
        lmb = 3.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_zero_lmb(self, xp, scp, dtype):
        data = _make_data((9, 14), xp, dtype)
        lmb = 0.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_empty(self, xp, scp, dtype):
        data = _make_data((0,), xp, dtype)
        lmb = 3
        result = scp.stats.boxcox_llf(lmb, data)
        if xp is numpy:
            assert type(result) is float
            return numpy.array(result)
        else:
            return result

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_lmb_neg(self, xp, scp, dtype):
        data = xp.array([198.0, 233.0, 233.0, 392.0])
        lmb = -45
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_lmb_neg2(self, xp, scp, dtype):
        data = _make_data((3, 5), xp, dtype)
        lmb = -3.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_empty_neg_lmb(self, xp, scp, dtype):
        data = _make_data((0,), xp, dtype)
        lmb = -1.0
        result = scp.stats.boxcox_llf(lmb, data)
        if xp is numpy:
            assert type(result) is float
            return numpy.array(result)
        else:
            return result

    @pytest.mark.skipif(
        not sys.platform.startswith('linux'),
        reason="Return value of scipy.stats.boxcox_llf has large error")
    @testing.with_requires('scipy>=1.13')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-12, rtol=rtol)
    def test_instability_around_zero(self, xp, scp):
        data = xp.asarray([2003, 1950, 1997, 2000, 2009], dtype=numpy.float64)
        return scp.stats.boxcox_llf(1e-8, data)

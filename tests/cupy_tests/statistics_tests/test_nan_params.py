from __future__ import annotations

import warnings
import numpy
import pytest

import cupy
from cupy import testing


@testing.with_requires('numpy>=1.22')
class TestNanParams:

    #
    # Test nanmin and nanmax
    #
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanmin_max_where_initial(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        where = xp.array([[[True, False, True, True],
                           [False, True, True, False],
                           [True, True, False, True]],
                          [[True, True, True, True],
                           [False, False, False, False],
                           [True, False, True, False]]])
        initial = 5 if dtype().kind in 'biu' else 5.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r1 = xp.nanmin(a, axis=1, where=where, initial=initial)
            r2 = xp.nanmax(a, axis=1, where=where, initial=initial)
        return r1, r2

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanmin_max_all_nan_where(self, xp, dtype):
        a = xp.array(
            [[xp.nan, 2.0, 3.0], [xp.nan, xp.nan, xp.nan]], dtype=dtype)
        where = xp.array([[True, False, True], [True, True, True]])
        initial = 10.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r1 = xp.nanmin(a, axis=1, where=where, initial=initial)
            r2 = xp.nanmax(a, axis=1, where=where, initial=initial)
        return r1, r2

    @testing.for_all_dtypes(no_complex=True)
    def test_nanmin_max_where_no_initial_raises(self, dtype):
        a = cupy.array([1, 2, 3], dtype=dtype)
        where = cupy.array([True, False, True])
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.nanmin(a, where=where)
            with pytest.raises(ValueError):
                xp.nanmax(a, where=where)

    #
    # Test nansum and nanprod
    #
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nansum_prod_where_initial(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        where = xp.array([[[True, False, True, True],
                           [False, True, True, False],
                           [True, True, False, True]],
                          [[True, True, True, True],
                           [False, False, False, False],
                           [True, False, True, False]]])
        initial = 10
        r1 = xp.nansum(a, axis=1, where=where, initial=initial)
        r2 = xp.nanprod(a, axis=1, where=where, initial=initial)
        return r1, r2

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nansum_prod_empty_initial(self, xp, dtype):
        a = xp.array([], dtype=dtype)
        initial = 10
        r1 = xp.nansum(a, initial=initial)
        r2 = xp.nanprod(a, initial=initial)
        return r1, r2

    #
    # Test nanmean, nanvar, nanstd
    #
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_nanmean_var_std_where(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        where = xp.array([[[True, False, True, True],
                           [False, True, True, False],
                           [True, True, False, True]],
                          [[True, True, True, True],
                           [False, False, False, False],
                           [True, False, True, False]]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r1 = xp.nanmean(a, axis=1, where=where)
            r2 = xp.nanvar(a, axis=1, where=where)
            r3 = xp.nanstd(a, axis=1, where=where)
        return r1, r2, r3

    #
    # Test nanargmin and nanargmax
    #
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_argmax_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        r1 = xp.nanargmin(a, axis=1, keepdims=True)
        r2 = xp.nanargmax(a, axis=1, keepdims=True)
        return r1, r2

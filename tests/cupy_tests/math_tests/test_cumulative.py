import unittest
import numpy
import cupy
from cupy import testing

class TestCumulativeSumProd(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_cumulative_sum(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        if hasattr(xp, 'cumulative_sum'):
            return xp.cumulative_sum(a, axis=1)
        else:
            if xp is numpy:
                return xp.cumsum(a, axis=1)
            return xp.cumulative_sum(a, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_cumulative_prod(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        if hasattr(xp, 'cumulative_prod'):
            return xp.cumulative_prod(a, axis=1)
        else:
            if xp is numpy:
                return xp.cumprod(a, axis=1)
            return xp.cumulative_prod(a, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_cumulative_sum_1d(self, xp, dtype):
        a = testing.shaped_random((5,), xp, dtype)
        if hasattr(xp, 'cumulative_sum'):
            return xp.cumulative_sum(a)
        else:
            if xp is numpy:
                return xp.cumsum(a)
            return xp.cumulative_sum(a)

    @testing.for_all_dtypes(no_bool=True)
    def test_cumulative_sum_error(self, dtype):
        # Test that cumulative_sum raises ValueError when axis is None and ndim > 1
        a = cupy.zeros((2, 2), dtype=dtype)
        with self.assertRaises(ValueError):
            cupy.cumulative_sum(a)

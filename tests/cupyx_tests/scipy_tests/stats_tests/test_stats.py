import numpy
import pytest

import cupy
from cupy import testing
import cupyx
import cupyx.scipy.stats  # NOQA

try:
    import scipy.stats
except ImportError:
    pass


@testing.with_requires('scipy')
class TestTrim:

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_base1(self, xp, scp, dtype, order):
        idx = xp.array([3, 5, 0, 1, 2, 4])
        a = testing.shaped_arange(
            (6, 4), xp=xp, dtype=dtype, order=order)[idx, :]
        return scp.stats.trim_mean(a, 2 / 6.)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_base2(self, xp, scp, dtype, order):
        idx = xp.array([1, 0, 3, 2])
        a = testing.shaped_arange(
            (4, 6), xp=xp, dtype=dtype, order=order)[idx, :]
        return scp.stats.trim_mean(a, 2 / 6.)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_array_like(self, xp, scp):
        a = xp.array([
            7, 11, 12, 21, 16, 6, 22, 1, 5, 0, 18, 10, 17, 9, 19, 15, 23, 20,
            2, 14, 4, 13, 8, 3
        ])
        return scp.stats.trim_mean(a, 2 / 6.)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_reversed(self, xp, scp):
        a = xp.array([5, 4, 3, 1, 2, 0])
        return scp.stats.trim_mean(a, 2 / 6.)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_axis(self, xp, scp, dtype):
        a = testing.shaped_arange((5, 6, 4, 7), xp=xp, dtype=dtype)
        return [scp.stats.trim_mean(a, 2 / 6., axis=axis)
                for axis in [0, 1, 2, 3, -1]]

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_axis_none(self, xp, scp):
        a = xp.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
        return scp.stats.trim_mean(a, 2 / 6., axis=None)

    def test_invalid_propotiontocut(self):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            a = xp.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
            with pytest.raises(ValueError):
                scp.stats.trim_mean(a, 0.6)

    @pytest.mark.parametrize('propotiontocut', [0.0, 0.6])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_empty_input(self, xp, scp, propotiontocut):
        return scp.stats.trim_mean(xp.array([]), propotiontocut)

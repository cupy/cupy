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
    @testing.numpy_cupy_allclose(
        scipy_name='scp', rtol=1e-6, contiguous_check=False)
    @pytest.mark.parametrize('shape', [(24,), (6, 4), (6, 4, 3), (4, 6)])
    def test_base(self, xp, scp, dtype, order, shape):
        a = testing.shaped_random(
            shape, xp=xp, dtype=dtype, order=order, scale=100)
        return scp.stats.trim_mean(a, 2 / 6.)

    @testing.for_all_dtypes()
    def test_zero_dim(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            a = xp.array(0, dtype=dtype)
            with pytest.raises(IndexError):
                return scp.stats.trim_mean(a, 2 / 6.)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_dim_axis_none(self, xp, scp, dtype):
        a = xp.array(0, dtype=dtype)
        return scp.stats.trim_mean(a, 2 / 6., axis=None)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('propotiontocut', [0.0, 0.6])
    def test_empty(self, xp, scp, dtype, propotiontocut):
        a = xp.array([])
        return scp.stats.trim_mean(a, 2 / 6., propotiontocut)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        scipy_name='scp', rtol=1e-6, contiguous_check=False)
    @pytest.mark.parametrize('axis', [0, 1, 2, 3, -1, None])
    def test_axis(self, xp, scp, dtype, order, axis):
        a = testing.shaped_random(
            (5, 6, 4, 7), xp=xp, dtype=dtype, order=order, scale=100)
        return scp.stats.trim_mean(a, 2 / 6., axis=axis)

    def test_propotion_too_big(self):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            a = xp.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
            with pytest.raises(ValueError):
                scp.stats.trim_mean(a, 0.6)

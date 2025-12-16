from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
import cupyx
import cupyx.scipy.stats  # NOQA
from cupyx.scipy.stats import rankdata
from cupy.testing import assert_array_equal


try:
    import scipy.stats
except ImportError:
    pass


atol = {'default': 1e-6, cupy.float64: 1e-14}
rtol = {'default': 1e-6, cupy.float64: 1e-14}


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


@testing.with_requires('scipy')
class TestZmap:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((10,), xp, dtype=dtype)
        y = testing.shaped_random((8,), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((2, 6), xp, dtype=dtype)
        y = testing.shaped_random((2, 1), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_multi_dim(self, xp, scp, dtype):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=dtype)
        y = testing.shaped_random((3, 4, 1, 1), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_multi_dim_2(self, xp, scp, dtype):
        x = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=dtype)
        y = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zmap_multi_dim_2_float16(self, xp, scp):
        x = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=xp.float16)
        y = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=xp.float16)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_with_axis(self, xp, scp, dtype):
        x = testing.shaped_random((2, 3), xp, dtype=dtype)
        y = testing.shaped_random((1, 3), xp, dtype=dtype)
        return scp.stats.zmap(x, y, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_with_axis_ddof(self, xp, scp, dtype):
        x = testing.shaped_random((4, 5), xp, dtype=dtype)
        y = testing.shaped_random((1, 5), xp, dtype=dtype)
        return scp.stats.zmap(x, y, axis=1, ddof=2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        y = xp.array([1, 3, 5], dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        y = xp.array([xp.nan, -4.0, -1.0, -5.0], dtype=dtype)
        with numpy.errstate(invalid='ignore'):  # numpy warns with complex
            return scp.stats.zmap(x, y, nan_policy='propagate')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        y = xp.array([xp.nan, -4.0, -1.0, -5.0], dtype=dtype)
        return scp.stats.zmap(x, y, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_omit_axis_ddof(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        y = xp.array([xp.nan, -4.0, -1.0, -5.0], dtype=dtype)
        return scp.stats.zmap(x, y, axis=0, ddof=1, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1, 2, 3], dtype=dtype)
            y = xp.array([8, -4, xp.nan, 4], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.zmap(x, y, nan_policy='raise')


@testing.with_requires('scipy')
class TestZscore:

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((10,), xp, dtype=dtype)
        return scp.stats.zscore(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-3)
    def test_zscore_1dim_float16(self, xp, scp):
        x = testing.shaped_random((10,), xp, dtype=xp.float16)
        return scp.stats.zscore(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((5, 3), xp, dtype=dtype)
        return scp.stats.zscore(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zscore_2dim_float16(self, xp, scp):
        x = testing.shaped_random((5, 3), xp, dtype=xp.float16)
        return scp.stats.zscore(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_multi_dim(self, xp, scp, dtype):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=dtype)
        return scp.stats.zscore(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zscore_multi_dim_float16(self, xp, scp):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=xp.float16)
        return scp.stats.zscore(x)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=atol)
    def test_zscore_with_axis(self, xp, scp, dtype):
        x = testing.shaped_random((5, 6), xp, dtype=dtype)
        return scp.stats.zscore(x, axis=1)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_with_axis_ddof(self, xp, scp, dtype):
        x = testing.shaped_random((2, 3, 8), xp, dtype=dtype)
        return scp.stats.zscore(x, axis=2, ddof=2)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zscore_with_axis_ddof_float16(self, xp, scp):
        x = testing.shaped_random((2, 3, 8), xp, dtype=xp.float16)
        return scp.stats.zscore(x, axis=2, ddof=2)

    @testing.with_requires('scipy>=1.15')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zscore_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        return scp.stats.zscore(x)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        with numpy.errstate(invalid='ignore'):  # numpy warns with complex
            return scp.stats.zscore(x, nan_policy='propagate')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        return scp.stats.zscore(x, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_omit_axis_ddof(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        return scp.stats.zscore(x, axis=0, ddof=1, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1, 2, 3, xp.nan], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.zscore(x, nan_policy='raise')


class TestRankData:

    def test_empty(self):
        """stats.rankdata([]) should return an empty array."""
        a = cupy.array([], dtype=int)
        r = rankdata(a)
        assert_array_equal(r, cupy.array([], dtype=cupy.float64))
        r = rankdata([])
        assert_array_equal(r, cupy.array([], dtype=cupy.float64))

    @pytest.mark.parametrize("shape", [(0, 1, 2)])
    @pytest.mark.parametrize("axis", [None, *range(3)])
    def test_empty_multidim(self, shape, axis):
        a = cupy.empty(shape, dtype=int)
        r = rankdata(a, axis=axis)
        expected_shape = (0,) if axis is None else shape
        assert_array_equal(r.shape, expected_shape)
        assert r.dtype == cupy.float64

    def test_one(self):
        """Check stats.rankdata with an array of length 1."""
        data = [100]
        a = cupy.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, cupy.array([1.0], dtype=cupy.float64))
        r = rankdata(data)
        assert_array_equal(r, cupy.array([1.0], dtype=cupy.float64))

    def test_basic(self):
        """Basic tests of stats.rankdata."""
        data = [100, 10, 50]
        expected = cupy.array([3.0, 1.0, 2.0], dtype=cupy.float64)
        a = cupy.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)

        data = [40, 10, 30, 10, 50]
        expected = cupy.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=cupy.float64)
        a = cupy.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)

        data = [20, 20, 20, 10, 10, 10]
        expected = cupy.array(
            [5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=cupy.float64)
        a = cupy.array(data, dtype=int)
        r = rankdata(a)
        assert_array_equal(r, expected)
        r = rankdata(data)
        assert_array_equal(r, expected)
        # The docstring states explicitly that the argument is flattened.
        a2d = a.reshape(2, 3)
        r = rankdata(a2d)
        assert_array_equal(r, expected)

    def test_large_int(self):
        data = cupy.array([2**60, 2**60+1], dtype=cupy.uint64)
        r = rankdata(data)
        assert_array_equal(r, [1.0, 2.0])

        data = cupy.array([2**60, 2**60+1], dtype=cupy.int64)
        r = rankdata(data)
        assert_array_equal(r, [1.0, 2.0])

        data = cupy.array([2**60, -2**60+1], dtype=cupy.int64)
        r = rankdata(data)
        assert_array_equal(r, [2.0, 1.0])

    def test_big_tie(self):
        for n in [10000, 100000, 1000000]:
            data = cupy.ones(n, dtype=int)
            r = rankdata(data)
            expected_rank = 0.5 * (n + 1)
            assert_array_equal(r, expected_rank * data,
                               err_msg=f"test failed with n={n}")

    def test_axis(self):
        data = [[0, 2, 1],
                [4, 2, 2]]
        expected0 = [[1., 1.5, 1.],
                     [2., 1.5, 2.]]
        r0 = rankdata(data, axis=0)
        assert_array_equal(r0, expected0)
        expected1 = [[1., 3., 2.],
                     [3., 1.5, 1.5]]
        r1 = rankdata(data, axis=1)
        assert_array_equal(r1, expected1)

    methods = ["average", "min", "max", "dense", "ordinal"]
    dtypes = [cupy.float64] + [cupy.int64]*4

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("method, dtype", zip(methods, dtypes))
    def test_size_0_axis(self, axis, method, dtype):
        shape = (3, 0)
        data = cupy.zeros(shape)
        r = rankdata(data, method=method, axis=axis)
        assert_array_equal(r.shape, shape)
        assert r.dtype == dtype

    @pytest.mark.parametrize('axis', range(3))
    @pytest.mark.parametrize('method', methods)
    def test_nan_policy_omit_3d(self, axis, method):
        shape = (20, 21, 22)
        rng = cupy.random.RandomState(23983242)

        a = rng.rand(*shape)
        i = rng.rand(*shape) < 0.4
        j = rng.rand(*shape) < 0.1
        k = rng.rand(*shape) < 0.1
        a[i] = cupy.nan
        a[j] = -cupy.inf
        a[k] - cupy.inf

        def rank_1d_omit(a, method):
            out = cupy.zeros_like(a)
            i = cupy.isnan(a)
            a_compressed = a[~i]
            res = rankdata(a_compressed, method)
            out[~i] = res
            out[i] = cupy.nan
            return out

        def rank_omit(a, method, axis):
            return cupy.apply_along_axis(lambda a: rank_1d_omit(a, method),
                                         axis, a)

        res = rankdata(a, method, axis=axis, nan_policy='omit')
        res0 = rank_omit(a, method, axis=axis)

        assert_array_equal(res, res0)

    def test_nan_policy_2d_axis_none(self):
        # 2 2d-array test with axis=None
        data = [[0, cupy.nan, 3],
                [4, 2, cupy.nan],
                [1, 2, 2]]
        assert_array_equal(rankdata(data, axis=None, nan_policy='omit'),
                           [1., cupy.nan, 6., 7., 4., cupy.nan, 2., 4., 4.])
        assert_array_equal(rankdata(data, axis=None, nan_policy='propagate'),
                           [cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan,
                            cupy.nan, cupy.nan, cupy.nan, cupy.nan])

    def test_nan_policy_propagate(self):
        # 1 1d-array test
        data = [0, 2, 3, -2, cupy.nan, cupy.nan]
        assert_array_equal(rankdata(data, nan_policy='propagate'),
                           [cupy.nan, cupy.nan, cupy.nan, cupy.nan, cupy.nan,
                            cupy.nan])

        # 2 2d-array test
        data = [[0, cupy.nan, 3],
                [4, 2, cupy.nan],
                [1, 2, 2]]
        assert_array_equal(rankdata(data, axis=0, nan_policy='propagate'),
                           [[1, cupy.nan, cupy.nan],
                            [3, cupy.nan, cupy.nan],
                            [2, cupy.nan, cupy.nan]])
        assert_array_equal(rankdata(data, axis=1, nan_policy='propagate'),
                           [[cupy.nan, cupy.nan, cupy.nan],
                            [cupy.nan, cupy.nan, cupy.nan],
                            [1, 2.5, 2.5]])


_cases = (
    # values, method, expected
    ([], 'average', []),
    ([], 'min', []),
    ([], 'max', []),
    ([], 'dense', []),
    ([], 'ordinal', []),
    #
    ([100], 'average', [1.0]),
    ([100], 'min', [1.0]),
    ([100], 'max', [1.0]),
    ([100], 'dense', [1.0]),
    ([100], 'ordinal', [1.0]),
    #
    ([100, 100, 100], 'average', [2.0, 2.0, 2.0]),
    ([100, 100, 100], 'min', [1.0, 1.0, 1.0]),
    ([100, 100, 100], 'max', [3.0, 3.0, 3.0]),
    ([100, 100, 100], 'dense', [1.0, 1.0, 1.0]),
    ([100, 100, 100], 'ordinal', [1.0, 2.0, 3.0]),
    #
    ([100, 300, 200], 'average', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'min', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'max', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'dense', [1.0, 3.0, 2.0]),
    ([100, 300, 200], 'ordinal', [1.0, 3.0, 2.0]),
    #
    ([100, 200, 300, 200], 'average', [1.0, 2.5, 4.0, 2.5]),
    ([100, 200, 300, 200], 'min', [1.0, 2.0, 4.0, 2.0]),
    ([100, 200, 300, 200], 'max', [1.0, 3.0, 4.0, 3.0]),
    ([100, 200, 300, 200], 'dense', [1.0, 2.0, 3.0, 2.0]),
    ([100, 200, 300, 200], 'ordinal', [1.0, 2.0, 4.0, 3.0]),
    #
    ([100, 200, 300, 200, 100], 'average', [1.5, 3.5, 5.0, 3.5, 1.5]),
    ([100, 200, 300, 200, 100], 'min', [1.0, 3.0, 5.0, 3.0, 1.0]),
    ([100, 200, 300, 200, 100], 'max', [2.0, 4.0, 5.0, 4.0, 2.0]),
    ([100, 200, 300, 200, 100], 'dense', [1.0, 2.0, 3.0, 2.0, 1.0]),
    ([100, 200, 300, 200, 100], 'ordinal', [1.0, 3.0, 5.0, 4.0, 2.0]),
    #
    ([10] * 30, 'ordinal', cupy.arange(1.0, 31.0)),
)


def test_cases():
    for values, method, expected in _cases:
        r = rankdata(values, method=method)
        assert_array_equal(r, expected)

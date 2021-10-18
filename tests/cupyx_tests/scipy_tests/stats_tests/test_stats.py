import unittest

import pytest

import cupy as cp
from cupy import testing
from cupyx.scipy import stats


@testing.gpu
class TestTrim(unittest.TestCase):
    # test trim functions

    def test_trim_mean(self):
        # don't use pre-sorted arrays
        a = cp.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
        idx = cp.array([3, 5, 0, 1, 2, 4])
        a2 = cp.arange(24).reshape(6, 4)[idx, :]
        a3 = cp.arange(24).reshape(6, 4, order='F')[idx, :]
        testing.assert_array_equal(stats.trim_mean(a3, 2 / 6.),
                                   cp.array([2.5, 8.5, 14.5, 20.5]))
        testing.assert_array_equal(stats.trim_mean(a2, 2 / 6.),
                                   cp.array([10., 11., 12., 13.]))
        idx4 = cp.array([1, 0, 3, 2])
        a4 = cp.arange(24).reshape(4, 6)[idx4, :]
        testing.assert_array_equal(stats.trim_mean(a4, 2 / 6.),
                                   cp.array([9., 10., 11., 12., 13., 14.]))
        # shuffled arange(24) as array_like
        a = [
            7, 11, 12, 21, 16, 6, 22, 1, 5, 0, 18, 10, 17, 9, 19, 15, 23, 20,
            2, 14, 4, 13, 8, 3
        ]
        testing.assert_array_equal(stats.trim_mean(a, 2 / 6.), 11.5)
        testing.assert_array_equal(stats.trim_mean([5, 4, 3, 1, 2, 0], 2 / 6.),
                                   2.5)

        # check axis argument
        cp.random.seed(1234)
        a = cp.random.randint(20, size=(5, 6, 4, 7))
        for axis in [0, 1, 2, 3, -1]:
            res1 = stats.trim_mean(a, 2 / 6., axis=axis)
            res2 = stats.trim_mean(cp.moveaxis(a, axis, 0), 2 / 6.)
            testing.assert_array_equal(res1, res2)

        res1 = stats.trim_mean(a, 2 / 6., axis=None)
        res2 = stats.trim_mean(a.ravel(), 2 / 6.)
        testing.assert_array_equal(res1, res2)

        pytest.raises(ValueError, stats.trim_mean, a, 0.6)

        # empty input
        testing.assert_array_equal(stats.trim_mean([], 0.0), cp.nan)
        testing.assert_array_equal(stats.trim_mean([], 0.6), cp.nan)

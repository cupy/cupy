import unittest

import numpy as np
import six

import cupy as cp
from cupy import cuda
from cupy import testing
from cupy.testing import condition
from cupy.testing.array import assert_array_equal
from cupy.lib.stride_tricks import as_strided


@testing.gpu
class TestAsStrided(unittest.TestCase):
    def test_as_strided(self):
        a = cp.array([1, 2, 3, 4])
        a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
        expected = cp.array([1, 3])
        assert_array_equal(a_view, expected)

        a = cp.array([1, 2, 3, 4])
        a_view = as_strided(a, shape=(3, 4), strides=(0, 1 * a.itemsize))
        expected = cp.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        assert_array_equal(a_view, expected)

    def test_rolling_window(self):
        for shape, window, axis in [((3, 4), 2, 0), ((10, 30, 4), 4, -2)]:
            a = np.random.randn(*shape)
            a_rw = rolling_window(a, window, axis)
            b = cp.array(a)
            b_rw = rolling_window(b, window, axis)
            assert_array_equal(a_rw, b_rw)


def rolling_window(a, window, axis=-1):
    """
    Make an ndarray with a rolling window along axis.
    This function is taken from https://github.com/numpy/numpy/pull/31
    but slightly modified to accept axis option.
    """
    a = np.swapaxes(a, axis, -1)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    if isinstance(a, np.ndarray):
        rolling = np.lib.stride_tricks.as_strided(
            a, shape=shape, strides=strides)
    elif isinstance(a, cp.ndarray):
        rolling = as_strided(a, shape=shape, strides=strides)
    return rolling.swapaxes(-2, axis)

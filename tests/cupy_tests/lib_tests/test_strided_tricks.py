import unittest

import numpy

import cupy
from cupy import testing
from cupy.lib import stride_tricks


@testing.gpu
class TestAsStrided(unittest.TestCase):
    def test_as_strided(self):
        a = cupy.array([1, 2, 3, 4])
        a_view = stride_tricks.as_strided(
            a, shape=(2,), strides=(2 * a.itemsize,))
        expected = cupy.array([1, 3])
        testing.assert_array_equal(a_view, expected)

        a = cupy.array([1, 2, 3, 4])
        a_view = stride_tricks.as_strided(
            a, shape=(3, 4), strides=(0, 1 * a.itemsize))
        expected = cupy.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        testing.assert_array_equal(a_view, expected)

    @testing.numpy_cupy_array_equal()
    def test_rolling_window(self, xp):
        a = testing.shaped_arange((3, 4), xp)
        a_rolling = rolling_window(a, 2, 0)

        return a_rolling


def rolling_window(a, window, axis=-1):
    """
    Make an ndarray with a rolling window along axis.
    This function is taken from https://github.com/numpy/numpy/pull/31
    but slightly modified to accept axis option.
    """
    a = numpy.swapaxes(a, axis, -1)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    if isinstance(a, numpy.ndarray):
        rolling = numpy.lib.stride_tricks.as_strided(
            a, shape=shape, strides=strides)
    elif isinstance(a, cupy.ndarray):
        rolling = stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolling.swapaxes(-2, axis)

import unittest

import numpy

import cupy
from cupy import testing
from cupy.lib import stride_tricks


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


class TestSlidingWindowView:
    def test_1d(self):
        arr = np.arange(5)
        arr_view = sliding_window_view(arr, 2)
        expected = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4]])
        assert_array_equal(arr_view, expected)

    def test_2d(self):
        i, j = np.ogrid[:3, :4]
        arr = 10*i + j
        shape = (2, 2)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1], [10, 11]],
                              [[1, 2], [11, 12]],
                              [[2, 3], [12, 13]]],
                             [[[10, 11], [20, 21]],
                              [[11, 12], [21, 22]],
                              [[12, 13], [22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_with_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10*i + j
        arr_view = sliding_window_view(arr, 3, 0)
        expected = np.array([[[0, 10, 20],
                              [1, 11, 21],
                              [2, 12, 22],
                              [3, 13, 23]]])
        assert_array_equal(arr_view, expected)

    def test_2d_repeated_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10*i + j
        arr_view = sliding_window_view(arr, (2, 3), (1, 1))
        expected = np.array([[[[0, 1, 2],
                               [1, 2, 3]]],
                             [[[10, 11, 12],
                               [11, 12, 13]]],
                             [[[20, 21, 22],
                               [21, 22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_without_axis(self):
        i, j = np.ogrid[:4, :4]
        arr = 10*i + j
        shape = (2, 3)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1, 2], [10, 11, 12]],
                              [[1, 2, 3], [11, 12, 13]]],
                             [[[10, 11, 12], [20, 21, 22]],
                              [[11, 12, 13], [21, 22, 23]]],
                             [[[20, 21, 22], [30, 31, 32]],
                              [[21, 22, 23], [31, 32, 33]]]])
        assert_array_equal(arr_view, expected)


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

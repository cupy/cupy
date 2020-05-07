import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 1},
    {'repeats': 2, 'axis': -1},
    {'repeats': [0, 0, 0], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': -2},
)
@testing.gpu
class TestRepeat(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


class TestRepeatRepeatsNdarray(unittest.TestCase):

    def test_func(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        repeats = cupy.array([2, 3], dtype=cupy.int32)
        with pytest.raises(ValueError, match=r'repeats'):
            cupy.repeat(a, repeats)

    def test_method(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        repeats = cupy.array([2, 3], dtype=cupy.int32)
        with pytest.raises(ValueError, match=r'repeats'):
            a.repeat(repeats)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 1},
)
@testing.gpu
class TestRepeatListBroadcast(unittest.TestCase):

    """Test for `repeats` argument using single element list.

    This feature is only supported in NumPy 1.10 or later.
    """

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 0},
    {'repeats': [1, 2, 3, 4], 'axis': None},
    {'repeats': [1, 2, 3, 4], 'axis': 0},
)
@testing.gpu
class TestRepeat1D(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 0},
)
@testing.gpu
class TestRepeat1DListBroadcast(unittest.TestCase):

    """See comment in TestRepeatListBroadcast class."""

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': -3, 'axis': None},
    {'repeats': [-3, -3], 'axis': 0},
    {'repeats': [1, 2, 3], 'axis': None},
    {'repeats': [1, 2], 'axis': 1},
    {'repeats': 2, 'axis': -4},
    {'repeats': 2, 'axis': 3},
)
@testing.gpu
class TestRepeatFailure(unittest.TestCase):

    def test_repeat_failure(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'reps': 0},
    {'reps': 1},
    {'reps': 2},
    {'reps': (0, 1)},
    {'reps': (2, 3)},
    {'reps': (2, 3, 4, 5)},
)
@testing.gpu
class TestTile(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_tile(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.tile(x, self.reps)


@testing.parameterize(
    {'reps': -1},
    {'reps': (-1, -2)},
)
@testing.gpu
class TestTileFailure(unittest.TestCase):

    def test_tile_failure(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.tile(x, -3)

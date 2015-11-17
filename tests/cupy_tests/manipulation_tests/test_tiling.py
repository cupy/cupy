import unittest

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

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': -3, 'axis': None},
    {'repeats': [-3, -3], 'axis': 0},
    {'repeats': [1, 2, 3], 'axis': None},
    {'repeats': [1, 2], 'axis': 1},
)
@testing.gpu
class TestRepeatFailure(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_raises()
    def test_repeat_failure(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        xp.repeat(x, -3)


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

    _multiprocess_can_split_ = True

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

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_raises()
    def test_tile_failure(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        xp.tile(x, -3)

import unittest

from cupy import testing


@testing.gpu
class TestRearrange(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=True)
    def test_roll(self, xp, dtype):
        x = xp.arange(10, dtype)
        return xp.roll(x, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll2(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_negative(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, -2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_with_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_with_negative_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=-1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_double_shift(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        return xp.roll(x, 35)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_double_shift_with_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 11, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_zero_array(self, xp, dtype):
        x = testing.shaped_arange((), xp, dtype)
        return xp.roll(x, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_roll_invalid_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_roll_invalid_negative_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=-3)

import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
@testing.parameterize(
    {'shape': (10,), 'shift': 2, 'axis': None},
    {'shape': (5, 2), 'shift': 1, 'axis': None},
    {'shape': (5, 2), 'shift': -2, 'axis': None},
    {'shape': (5, 2), 'shift': 1, 'axis': 0},
    {'shape': (5, 2), 'shift': 1, 'axis': -1},
    {'shape': (10,), 'shift': 35, 'axis': None},
    {'shape': (5, 2), 'shift': 11, 'axis': 0},
    {'shape': (), 'shift': 5, 'axis': None},
    {'shape': (5, 2), 'shift': 1, 'axis': (0, 1)},
    {'shape': (5, 2), 'shift': 1, 'axis': (0, 0)},
    {'shape': (5, 2), 'shift': 50, 'axis': 0},
    {'shape': (5, 2), 'shift': (2, 1), 'axis': (0, 1)},
    {'shape': (5, 2), 'shift': (2, 1), 'axis': (0, -1)},
    {'shape': (5, 2), 'shift': (2, 1), 'axis': (1, -1)},
    {'shape': (5, 2), 'shift': (2, 1, 3), 'axis': 0},
    {'shape': (5, 2), 'shift': (2, 1, 3), 'axis': None},
)
class TestRoll(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        return xp.roll(x, self.shift, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_cupy_shift(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        shift = self.shift
        if xp is cupy:
            shift = cupy.array(shift)
        return xp.roll(x, shift, axis=self.axis)


class TestRollTypeError(unittest.TestCase):

    def test_roll_invalid_shift(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(TypeError):
                xp.roll(x, '0', axis=0)

    def test_roll_invalid_axis_type(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(TypeError):
                xp.roll(x, 2, axis='0')


@testing.parameterize(
    {'shape': (5, 2, 3), 'shift': (2, 2, 2), 'axis': (0, 1)},
    {'shape': (5, 2), 'shift': 1, 'axis': 2},
    {'shape': (5, 2), 'shift': 1, 'axis': -3},
    {'shape': (5, 2, 2), 'shift': (1, 0), 'axis': (0, 1, 2)},
    {'shape': (5, 2), 'shift': 1, 'axis': -3},
    {'shape': (5, 2), 'shift': 1, 'axis': (1, -3)},
)
class TestRollValueError(unittest.TestCase):
    def test_roll_invalid(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange(self.shape, xp)
            with pytest.raises(ValueError):
                xp.roll(x, self.shift, axis=self.axis)

    def test_roll_invalid_cupy_shift(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange(self.shape, xp)
            shift = self.shift
            if xp is cupy:
                shift = cupy.array(shift)
            with pytest.raises(ValueError):
                xp.roll(x, shift, axis=self.axis)


@testing.gpu
class TestFliplr(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fliplr_2(self, xp, dtype):
        x = testing.shaped_arange((3, 4), xp, dtype)
        return xp.fliplr(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fliplr_3(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.fliplr(x)

    @testing.for_all_dtypes()
    def test_fliplr_insufficient_ndim(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3,), xp, dtype)
            with pytest.raises(ValueError):
                xp.fliplr(x)


@testing.gpu
class TestFlipud(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flipud_1(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.flipud(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flipud_2(self, xp, dtype):
        x = testing.shaped_arange((3, 4), xp, dtype)
        return xp.flipud(x)

    @testing.for_all_dtypes()
    def test_flipud_insufficient_ndim(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((), xp, dtype)
            with pytest.raises(ValueError):
                xp.flipud(x)


@testing.gpu
class TestFlip(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_1(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.flip(x, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_2(self, xp, dtype):
        x = testing.shaped_arange((3, 4), xp, dtype)
        return xp.flip(x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_all(self, xp, dtype):
        x = testing.shaped_arange((3, 4), xp, dtype)
        return xp.flip(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_with_negative_axis(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.flip(x, -1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_with_axis_tuple(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.flip(x, (0, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_empty_dim_1(self, xp, dtype):
        x = xp.array([], dtype).reshape((0,))
        return xp.flip(x, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_empty_dim_2(self, xp, dtype):
        x = xp.array([], dtype).reshape((0, 0))
        return xp.flip(x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_empty_dim_3(self, xp, dtype):
        x = xp.array([], dtype).reshape((1, 0, 1))
        return xp.flip(x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flip_empty_dim_all(self, xp, dtype):
        x = xp.array([], dtype).reshape((1, 0, 1))
        return xp.flip(x)

    @testing.for_all_dtypes()
    def test_flip_insufficient_ndim(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((), xp, dtype)
            with pytest.raises(ValueError):
                xp.flip(x, 0)

    @testing.for_all_dtypes()
    def test_flip_invalid_axis(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.flip(x, 2)

    @testing.for_all_dtypes()
    def test_flip_invalid_negative_axis(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.flip(x, -3)


@testing.gpu
class TestRot90(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_rot90_none(self, xp, dtype):
        x = testing.shaped_arange((3, 4), xp, dtype)
        return xp.rot90(x, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_rot90_twice(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.rot90(x, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_rot90_negative(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.rot90(x, -1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_rot90_with_axes(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.rot90(x, 1, axes=(1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_rot90_with_negative_axes(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.rot90(x, 1, axes=(1, -1))

    @testing.for_all_dtypes()
    def test_rot90_insufficient_ndim(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3,), xp, dtype)
            with pytest.raises(ValueError):
                xp.rot90(x)

    @testing.for_all_dtypes()
    def test_rot90_too_much_axes(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4, 2), xp, dtype)
            with pytest.raises(ValueError):
                xp.rot90(x, 1, axes=(0, 1, 2))

    @testing.for_all_dtypes()
    def test_rot90_invalid_axes(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4, 2), xp, dtype)
            with pytest.raises(ValueError):
                xp.rot90(x, 1, axes=(1, 3))

    @testing.for_all_dtypes()
    def test_rot90_invalid_negative_axes(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4, 2), xp, dtype)
            with pytest.raises(ValueError):
                xp.rot90(x, 1, axes=(1, -2))

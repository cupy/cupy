import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestRoll(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
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
    @testing.numpy_cupy_array_equal()
    def test_roll_scalar_shift_multi_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=(0, 1))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_scalar_shift_duplicate_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 1, axis=(0, 0))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_large_shift(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, 50, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_multi_shift_multi_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1), axis=(0, 1))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_multi_shift_multi_axis2(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1), axis=(0, -1))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_multi_shift_multi_axis3(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1), axis=(1, -1))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_multi_shift_scalar_axis(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1, 3), axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_multi_shift_axis_none(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype)
        return xp.roll(x, (2, 1, 3), axis=None)

    def test_roll_invalid_shift(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(TypeError):
                xp.roll(x, '0', axis=0)

    def test_roll_shape_mismatch(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2, 3), xp)
            with pytest.raises(ValueError):
                xp.roll(x, (2, 2, 2), axis=(0, 1))

    def test_roll_invalid_axis1(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(ValueError):
                xp.roll(x, 1, axis=2)

    def test_roll_invalid_axis2(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(ValueError):
                xp.roll(x, 1, axis=-3)

    def test_roll_invalid_axis_length(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2, 2), xp)
            with pytest.raises(ValueError):
                cupy.roll(x, shift=(1, 0), axis=(0, 1, 2))

    def test_roll_invalid_axis_type(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(TypeError):
                xp.roll(x, 2, axis='0')

    def test_roll_invalid_negative_axis1(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(ValueError):
                xp.roll(x, 1, axis=-3)

    def test_roll_invalid_negative_axis2(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(ValueError):
                xp.roll(x, 1, axis=(1, -3))


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
    def test_flip_with_negative_axis(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.flip(x, -1)

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

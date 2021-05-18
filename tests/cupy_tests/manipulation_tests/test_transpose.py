import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestTranspose(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_moveaxis1(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 1], [1, 2])

    @testing.numpy_cupy_array_equal()
    def test_moveaxis2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, 1, -1)

    @testing.numpy_cupy_array_equal()
    def test_moveaxis3(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [0, 2], [1, 0])

    @testing.numpy_cupy_array_equal()
    def test_moveaxis4(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [2, 0], [1, 0])

    @testing.numpy_cupy_array_equal()
    def test_moveaxis5(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.moveaxis(a, [2, 0], [0, 1])

    @testing.numpy_cupy_array_equal()
    def test_moveaxis6(self, xp):
        a = testing.shaped_arange((2, 3, 4, 5, 6), xp)
        return xp.moveaxis(a, [0, 2, 1], [3, 4, 0])

    # dim is too large
    def test_moveaxis_invalid1_1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, [0, 1], [1, 3])

    def test_moveaxis_invalid1_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, [0, 1], [1, 3])

    def test_moveaxis_invalid1_3(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, 0, 3)

    # dim is too small
    def test_moveaxis_invalid2_1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, [0, -4], [1, 2])

    def test_moveaxis_invalid2_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, [0, -4], [1, 2])

    def test_moveaxis_invalid2_3(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, -4, 0)

    # len(source) != len(destination)
    def test_moveaxis_invalid3_1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.moveaxis(a, [0, 1, 2], [1, 2])

    def test_moveaxis_invalid3_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.moveaxis(a, 0, [1, 2])

    # len(source) != len(destination)
    def test_moveaxis_invalid4_1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.moveaxis(a, [0, 1], [1, 2, 0])

    def test_moveaxis_invalid4_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.moveaxis(a, [0, 1], 1)

    # Use the same axis twice
    def test_moveaxis_invalid5_1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.moveaxis(a, [1, -1], [1, 3])

    def test_moveaxis_invalid5_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.moveaxis(a, [0, 1], [-1, 2])

    def test_moveaxis_invalid5_3(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.moveaxis(a, [0, 1], [1, 1])

    @testing.numpy_cupy_array_equal()
    def test_rollaxis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.rollaxis(a, 2)

    def test_rollaxis_failure(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.rollaxis(a, 3)

    @testing.numpy_cupy_array_equal()
    def test_swapaxes(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.swapaxes(a, 2, 0)

    def test_swapaxes_failure(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.swapaxes(a, 3, 0)

    @testing.numpy_cupy_array_equal()
    def test_transpose(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose(-1, 0, 1)

    @testing.numpy_cupy_array_equal()
    def test_transpose_empty(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose()

    @testing.numpy_cupy_array_equal()
    def test_transpose_none(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.transpose(None)

    @testing.numpy_cupy_array_equal()
    def test_external_transpose(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.transpose(a, (-1, 0, 1))

    @testing.numpy_cupy_array_equal()
    def test_external_transpose_all(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.transpose(a)

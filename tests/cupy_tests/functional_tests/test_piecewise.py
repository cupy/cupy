import unittest

import pytest

import cupy
from cupy import testing


class TestPiecewise(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_piecewise(self, xp, dtype):
        x = xp.linspace(2.5, 12.5, 6, dtype=dtype)
        condlist = [x < 0, x >= 0, x < 5, x >= 1.5]
        funclist = [-1, 1, 2, 5]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scalar_input(self, xp, dtype):
        x = dtype(2)
        condlist = [x < 0, x >= 0]
        funclist = [-10, 10]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scalar_condition(self, xp, dtype):
        x = testing.shaped_random(shape=(2, 3, 5), xp=xp, dtype=dtype)
        condlist = True
        funclist = [-10, 10]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_signed_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_otherwise_condition1(self, xp, dtype):
        x = xp.linspace(-2, 20, 12, dtype=dtype)
        condlist = [x > 15, x <= 5, x == 0, x == 10]
        funclist = [-1, 0, 2, 3, -5]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_otherwise_condition2(self, xp, dtype):
        x = cupy.array([-10, 20, 30, 40], dtype=dtype)
        condlist = [[True, False, False, True], [True, False, False, True]]
        funclist = [-1, 1, 2]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_piecewise_zero_dim(self, xp, dtype):
        x = testing.empty(xp=xp, dtype=dtype)
        condlist = [x < 0, x > 0]
        funclist = [-1, 1, 2]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_piecewise_ndim(self, xp, dtype):
        x = testing.shaped_random(shape=(2, 3, 5), xp=xp, dtype=dtype)
        condlist = [x < 0, x > 0]
        funclist = [-1, 1, 2]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_piecewise_ndarray(self, xp, dtype):
        x = xp.linspace(1, 20, 12, dtype=dtype)
        condlist = xp.array([x > 15, x <= 5, x == 0, x == 10])
        funclist = xp.array([-1, 0, 2, 3, -5], dtype=dtype)
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    def test_mismatched_lengths(self, dtype):
        x = cupy.linspace(-2, 4, 6, dtype=dtype)
        condlist = [x < 0, x >= 0]
        funclist = [-1, 0, 2, 4, 5]
        with pytest.raises(ValueError):
            cupy.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    def test_callable_funclist(self, dtype):
        x = cupy.linspace(-2, 4, 6, dtype=dtype)
        condlist = [x < 0, x > 0]
        funclist = [lambda x: -x, lambda x: x]
        with pytest.raises(NotImplementedError):
            cupy.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes()
    def test_mixed_funclist(self, dtype):
        x = cupy.linspace(-2, 2, 6, dtype=dtype)
        condlist = [x < 0, x == 0, x > 0]
        funclist = [-10, lambda x: -x, 10, lambda x: x]
        with pytest.raises(NotImplementedError):
            cupy.piecewise(x, condlist, funclist)

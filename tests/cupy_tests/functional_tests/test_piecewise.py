import unittest

import pytest

import cupy
from cupy import testing


class TestPiecewise(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_linespace(self, xp, dtype):
        x = xp.linspace(-2.5, 2.5, 6, dtype=dtype)
        condlist = [x < 0, x >= 0]
        funclist = [-1, 1]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_scalar_value(self, xp, dtype):
        x = dtype(2)
        condlist = [x < 0, x >= 0]
        funclist = [-10, 10]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_scalar_condition(self, xp, dtype):
        x = xp.linspace(-2.5, 2.5, 4, dtype=dtype)
        condlist = True
        funclist = [-10, 10]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_otherwise_condition1(self, xp, dtype):
        x = xp.linspace(-2, 4, 4, dtype=dtype)
        condlist = [x < 0, x >= 0]
        funclist = [-1, 0, 2]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_otherwise_condition2(self, xp, dtype):
        x = cupy.array([-10, 20, 30, 40], dtype=dtype)
        condlist = [[True, False, False, True], [True, False, False, True]]
        funclist = [-1, 1, 2]
        return xp.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_mismatched_lengths(self, dtype):
        x = cupy.linspace(-2, 4, 6, dtype=dtype)
        condlist = [x < 0, x >= 0]
        funclist = [-1, 0, 2, 4, 5]
        with pytest.raises(ValueError):
            cupy.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_callable_funclist(self, dtype):
        x = cupy.linspace(-2, 4, 6, dtype=dtype)
        condlist = [x < 0, x > 0]
        funclist = [lambda x: -x, lambda x: x]
        with pytest.raises(ValueError):
            cupy.piecewise(x, condlist, funclist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_mixed_funclist(self, dtype):
        x = cupy.linspace(-2, 2, 6, dtype=dtype)
        condlist = [x < 0, x == 0, x > 0]
        funclist = [-10, lambda x: -x, 10, lambda x: x]
        with pytest.raises(ValueError):
            cupy.piecewise(x, condlist, funclist)

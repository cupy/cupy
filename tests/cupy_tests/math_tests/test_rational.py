import unittest

import pytest

import cupy
from cupy import testing


@testing.gpu
class TestRational(unittest.TestCase):

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_gcd_basic(self, xp, dtype):
        a = testing.shaped_random((6, 6), xp, dtype)
        b = testing.shaped_random((6, 6), xp, dtype)
        return xp.gcd(a, b)

    @testing.for_float_dtypes()
    def test_gcd_dtype_check(self, dtype):
        a = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        b = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        with pytest.raises(TypeError):
            cupy.gcd(a, b)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_lcm_basic(self, xp, dtype):
        a = testing.shaped_random((6, 6), xp, dtype)
        b = testing.shaped_random((6, 6), xp, dtype)
        return xp.lcm(a, b)

    @testing.for_float_dtypes()
    def test_lcm_dtype_check(self, dtype):
        a = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        b = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        with pytest.raises(TypeError):
            cupy.lcm(a, b)

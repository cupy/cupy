import unittest

import pytest

import cupy
from cupy import testing


@testing.gpu
class TestRational(unittest.TestCase):

    @testing.for_dtypes(['?', 'e', 'f', 'd', 'F', 'D'])
    def test_gcd_dtype_check(self, dtype):
        a = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        b = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        with pytest.raises(TypeError):
            cupy.gcd(a, b)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_gcd_check_boundary_cases(self, xp, dtype):
        a = xp.array([0, -10, -5, 10, 410, 1, 6, 33])
        b = xp.array([0, 5, -10, -5, 20, 51, 6, 42])
        return xp.gcd(a, b)

    @testing.for_dtypes(['?', 'e', 'f', 'd', 'F', 'D'])
    def test_lcm_dtype_check(self, dtype):
        a = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        b = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        with pytest.raises(TypeError):
            cupy.lcm(a, b)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_lcm_check_boundary_cases(self, xp, dtype):
        a = xp.array([0, -10, -5, 10, 410, 1, 6, 33])
        b = xp.array([0, 5, -10, -5, 20, 51, 6, 42])
        return xp.lcm(a, b)

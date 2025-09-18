from __future__ import annotations

import unittest

from cupy import testing


class TestElementwise(unittest.TestCase):

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def check_unary_int(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 0, 1, 2, 3]).astype(dtype)
        return getattr(xp, name)(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def check_binary_int(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 0, 1, 2, 3]).astype(dtype)
        b = xp.array([0, 1, 2, 3, 4, 5, 6]).astype(dtype)
        return getattr(xp, name)(a, b)

    def test_bitwise_and(self):
        self.check_binary_int('bitwise_and')

    def test_bitwise_or(self):
        self.check_binary_int('bitwise_or')

    def test_bitwise_xor(self):
        self.check_binary_int('bitwise_xor')

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_bitwise_count(self, xp, dtype):
        import numpy as np
        info = np.iinfo(dtype)
        if np.issubdtype(dtype, np.signedinteger):
            obj = np.array([
                0, -1, 1, info.min, info.min + 1,
                info.max, info.max - 1, info.max // 2,
            ], dtype=dtype)
        else:
            obj = np.array([
                0, 1, info.max, info.max - 1, info.max // 2,
            ], dtype=dtype)
        a = xp.asarray(obj, dtype=dtype)
        self.check_unary_int('bitwise_count')
        return xp.bitwise_count(a)

    def test_invert(self):
        self.check_unary_int('invert')

    def test_left_shift(self):
        self.check_binary_int('left_shift')

    def test_right_shift(self):
        self.check_binary_int('right_shift')

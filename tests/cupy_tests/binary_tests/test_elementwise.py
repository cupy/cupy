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

    def test_bitwise_count(self):
        import cupy as _cp
        x = _cp.array([0, 1, 2, 3, 255], dtype=_cp.uint8)
        assert (_cp.bit_count(x) == _cp.bitwise_count(x)).all()
        import numpy as _np
        if hasattr(_np, 'bit_count'):
            self.check_unary_int('bit_count')
        else:
            self.check_unary_int('bitwise_count')


    def test_invert(self):
        self.check_unary_int('invert')

    def test_left_shift(self):
        self.check_binary_int('left_shift')

    def test_right_shift(self):
        self.check_binary_int('right_shift')

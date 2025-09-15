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
        @testing.numpy_cupy_array_equal()
        def _check(xp):
            import numpy as _np
            import cupy as _cp
            a = xp.array([-3, -2, -1, 0, 1, 2, 3]).astype(xp.int32)
            if xp is _np:
                if hasattr(_np, 'bit_count'):
                    return _np.bit_count(a)
                # fallback for older NumPy: view as unsigned and count bits
                width = a.dtype.itemsize * 8
                mask = (1 << width) - 1
                ua = a.astype(_np.uint32, copy=False).view({
                    1: _np.uint8, 2: _np.uint16, 4: _np.uint32, 8: _np.uint64
                }[a.dtype.itemsize])
                # use Python int.bit_count elementwise
                return _np.fromiter(((int(v) & mask).bit_count() for v in a.ravel()),
                                    dtype=_np.uint8).reshape(a.shape)
            else:
                return _cp.bitwise_count(a)
        _check()

    def test_bit_count_alias(self):
        @testing.numpy_cupy_array_equal()
        def _check(xp):
            import numpy as _np
            import cupy as _cp
            a = xp.array([0, 1, 2, 3, 255]).astype(xp.uint8)
            if xp is _np:
                if hasattr(_np, 'bit_count'):
                    return _np.bit_count(a)
                return _np.fromiter((int(v).bit_count() for v in a.ravel()),
                                    dtype=_np.uint8).reshape(a.shape)
            else:
                return _cp.bit_count(a)
        _check()

    def test_invert(self):
        self.check_unary_int('invert')

    def test_left_shift(self):
        self.check_binary_int('left_shift')

    def test_right_shift(self):
        self.check_binary_int('right_shift')

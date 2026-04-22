from __future__ import annotations

import unittest
import numpy
from cupy import testing


@testing.parameterize(*testing.product({
    'shape_pair': [
        ((5,), (5,)),
        ((2, 5), (2, 5)),
        ((2, 3, 5), (2, 3, 5)),
        ((2, 5), (5,)),  # broadcasting
        ((5,), (2, 5)),  # broadcasting
        ((2, 3, 5), (3, 5)),  # broadcasting
        ((3, 5), (2, 3, 5)),  # broadcasting
    ],
    'axis': [-1, 0],
}))
class TestVecdot(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, type_check=False)
    def test_vecdot(self, xp, dtype):
        shape1, shape2 = self.shape_pair
        axis = self.axis

        # Ensure axis is valid for both shapes
        if axis < -len(shape1) or axis >= len(shape1) or \
           axis < -len(shape2) or axis >= len(shape2):
            return xp.array(0)

        # Also ensure the core dimension size matches
        if shape1[axis] != shape2[axis]:
            return xp.array(0)

        a1 = testing.shaped_random(shape1, xp, dtype)
        a2 = testing.shaped_random(shape2, xp, dtype)

        try:
            return xp.vecdot(a1, a2, axis=axis)
        except (ValueError, AttributeError):
            if xp is numpy and not hasattr(numpy, 'vecdot'):
                # fallback for old numpy
                a1_conj = a1.conj() if a1.dtype.kind == 'c' else a1
                # vecdot broadcasting is a bit complex to mock perfectly with
                # sum(a*b) but for these test cases it should work if shapes
                # are same after moving axis
                a1_m = xp.moveaxis(a1_conj, axis, -1)
                a2_m = xp.moveaxis(a2, axis, -1)
                return (a1_m * a2_m).sum(axis=-1)
            raise


class TestVecdotSimple(unittest.TestCase):
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_vecdot_simple(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        b = xp.array([4, 5, 6], dtype=dtype)
        if hasattr(xp, 'vecdot'):
            return xp.vecdot(a, b)
        else:
            if xp is numpy:
                a_conj = a.conj() if a.dtype.kind == 'c' else a
                return (a_conj * b).sum()
            return xp.vecdot(a, b)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_vecdot_complex(self, xp, dtype):
        if xp.dtype(dtype).kind != 'c':
            return xp.array(0)
        a = xp.array([1+1j, 2+2j], dtype=dtype)
        b = xp.array([3+3j, 4+4j], dtype=dtype)
        if hasattr(xp, 'vecdot'):
            return xp.vecdot(a, b)
        else:
            if xp is numpy:
                return (a.conj() * b).sum()
            return xp.vecdot(a, b)

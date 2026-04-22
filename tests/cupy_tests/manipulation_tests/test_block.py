from __future__ import annotations
import unittest
import cupy
from cupy import testing


class TestBlock(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_block_2d(self, xp, dtype):
        a = testing.shaped_random((2, 2), xp, dtype)
        b = testing.shaped_random((2, 3), xp, dtype)
        c = testing.shaped_random((3, 2), xp, dtype)
        d = testing.shaped_random((3, 3), xp, dtype)
        return xp.block([[a, b], [c, d]])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_block_1d_to_2d(self, xp, dtype):
        a = testing.shaped_random((5,), xp, dtype)
        b = testing.shaped_random((5,), xp, dtype)
        return xp.block([a, b])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_block_3d(self, xp, dtype):
        a = testing.shaped_random((2, 2, 2), xp, dtype)
        b = testing.shaped_random((2, 2, 2), xp, dtype)
        return xp.block([[[a]], [[b]]])

    def test_block_empty(self):
        with self.assertRaises(ValueError):
            cupy.block([])

    def test_block_depth_mismatch(self):
        a = cupy.eye(2)
        # depth mismatch: [[a, a], a] is invalid in numpy.block
        # but [ [a, a], [a] ] is valid.
        with self.assertRaises(ValueError):
            cupy.block([[a, a], a])

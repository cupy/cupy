import unittest

import numpy as np

import cupy
from cupy import testing


@testing.gpu
class TestArrayUfunc(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_unary_op(self, dtype):
        a = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = np.sin(a)
        # numpy operation produced a cupy array
        self.assertTrue(isinstance(outa, cupy.ndarray))
        b = a.get()
        outb = np.sin(b)
        self.assertTrue(np.allclose(outa, outb))

    @testing.for_all_dtypes()
    def test_binary_op(self, op, xp, dtype):
        a1 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = np.add(a1, a2)
        # numpy operation produced a cupy array
        self.assertTrue(isinstance(outa, cupy.ndarray))
        b1 = a1.get()
        b2 = a2.get()
        outb = np.add(b1, b2)
        self.assertTrue(np.allclose(outa, outb))

    @testing.for_all_dtypes()
    def test_binary_mixed_op(self, op, xp, dtype):
        a1 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(np.array([0, 1, 2]), dtype=dtype).get()
        with self.assertRaises(TypeError):
            # attempt to add cupy and numpy arrays
            np.add(a1, a2)

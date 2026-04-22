import unittest
import numpy
import cupy
from cupy import testing

class TestMatrixTranspose(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_transpose(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        if hasattr(xp, 'matrix_transpose'):
            return xp.matrix_transpose(a)
        else:
            if xp is numpy:
                return xp.swapaxes(a, -1, -2)
            return xp.matrix_transpose(a)

    def test_matrix_transpose_error(self):
        a = cupy.zeros((5,))
        with self.assertRaises(ValueError):
            cupy.matrix_transpose(a)

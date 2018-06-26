import unittest

import cupy
from cupy import testing


@testing.gpu
class TestArrayUfunc(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_indexing(self, xp):
        a = cupy.testing.shaped_arange((3, 1), xp)[:, :, None]
        b = cupy.testing.shaped_arange((3, 2), xp)[:, None, :]
        return a * b

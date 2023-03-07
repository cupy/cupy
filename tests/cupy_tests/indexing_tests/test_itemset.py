import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (8, 2), 'index': 4, 'value': 7},
    {'shape': (25, 25), 'index': (15, 23), 'value': 19},
    {'shape': (3, 256, 256), 'index': (500), 'value': 255},
    {'shape': (3, 256, 256), 'index': (0, 100, 200), 'value': 5},
    {'shape': (20, 3, 128, 128), 'index': (560), 'value': 255},
    {'shape': (20, 3, 128, 128), 'index': (2, 0, 19, 34), 'value': 5},
)
@testing.gpu
class TestItemset(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_itemset(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        a.itemset(self.index, self.value)
        return a

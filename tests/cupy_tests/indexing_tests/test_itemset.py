import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (8, 2), 'index': 4, 'value': 7},
    {'shape': (25, 25), 'index': (15, 23), 'value': 19},
    {'shape': (1920, 1080), 'index': (5600), 'value': 255},
    {'shape': (3, 7), 'index': (0, 6), 'value': 5},
)
@testing.gpu
class TestItemset(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_itemset(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        a.itemset(self.index, self.value)
        return a

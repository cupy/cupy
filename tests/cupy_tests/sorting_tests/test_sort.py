import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestSort(unittest.TestCase):

    _multiprocess_can_split_ = True

    # Test ranks

    @testing.numpy_cupy_raises()
    def test_sort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        a.sort()

    def test_sort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(ValueError):
            a.sort()

    # Test dtypes

    # TODO(takagi): Test 'bqBHILQ' dtypes
    # TODO(takagi): Test numpy.float16
    # TODO(takagi): Test numpy.bool_
    @testing.for_dtypes(['h', 'i', 'l', numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose()
    def test_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a.sort()
        return a

    # Test views

    def test_sort_view(self):
        a = testing.shaped_random((10,), cupy)[::]  # with making a view
        with self.assertRaises(ValueError):
            a.sort()

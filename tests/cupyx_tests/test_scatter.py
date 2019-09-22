import unittest

import numpy

import cupy
from cupy import testing
import cupyx


@testing.gpu
class TestScatter(unittest.TestCase):

    def test_scatter_add(self):
        a = cupy.zeros((3,), dtype=numpy.float32)
        i = cupy.array([1, 1], numpy.int32)
        v = cupy.array([2., 1.], dtype=numpy.float32)
        cupyx.scatter_add(a, i, v)
        testing.assert_array_equal(a, cupy.array([0, 3, 0]))

    def test_scatter_max(self):
        a = cupy.zeros((4,), dtype=numpy.float32)
        i = cupy.array([0, 1, 0, 1, 2, 2], numpy.int32)
        v = cupy.array([-1, 1, 1, 3, 2, 4], dtype=numpy.float32)
        cupyx.scatter_max(a, i, v)
        testing.assert_array_equal(a, cupy.array([1, 3, 4, 0]))

    def test_scatter_min(self):
        a = cupy.zeros((4,), dtype=numpy.float32)
        i = cupy.array([0, 1, 0, 1, 2, 2], numpy.int32)
        v = cupy.array([1, -1, -1, -3, -2, -4], dtype=numpy.float32)
        cupyx.scatter_min(a, i, v)
        testing.assert_array_equal(a, cupy.array([-1, -3, -4, 0]))

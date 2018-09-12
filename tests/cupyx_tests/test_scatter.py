import unittest

import numpy

import cupy
from cupy import testing
import cupyx


@testing.gpu
@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
class TestScatter(unittest.TestCase):

    def test_scatter_add(self):
        a = cupy.zeros((3,), dtype=self.dtype)
        i = cupy.array([1, 1], numpy.int32)
        v = cupy.array([2., 1.], dtype=self.dtype)
        cupyx.scatter_add(a, i, v)
        testing.assert_array_equal(a, cupy.array([0, 3, 0]))

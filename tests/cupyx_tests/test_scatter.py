import unittest

import numpy

import cupy
from cupy import testing
from cupy.cuda import runtime
import cupyx


@testing.gpu
class TestScatter(unittest.TestCase):

    # HIP does not support fp16 atomicAdd
    @testing.for_dtypes('iILQfd' if runtime.is_hip else 'iILQefd')
    def test_scatter_add(self, dtype):
        a = cupy.zeros((3,), dtype=dtype)
        i = cupy.array([1, 1], numpy.int32)
        v = cupy.array([2., 1.], dtype=dtype)
        cupyx.scatter_add(a, i, v)
        testing.assert_array_equal(a, cupy.array([0, 3, 0], dtype=dtype))

    @testing.for_dtypes('iILQfd')
    def test_scatter_max(self, dtype):
        a = cupy.zeros((4,), dtype=dtype)
        i = cupy.array([0, 1, 0, 1, 2, 2], numpy.int32)
        v = cupy.array([0, 1, 1, 3, 2, 4], dtype=dtype)
        cupyx.scatter_max(a, i, v)
        testing.assert_array_equal(a, cupy.array([1, 3, 4, 0], dtype=dtype))

    @testing.for_dtypes('iILQfd')
    def test_scatter_min(self, dtype):
        a = cupy.full((4,), 10, dtype=dtype)
        i = cupy.array([0, 1, 0, 1, 2, 2], numpy.int32)
        v = cupy.array([6, 4, 4, 2, 3, 1], dtype=dtype)
        cupyx.scatter_min(a, i, v)
        testing.assert_array_equal(a, cupy.array([4, 2, 1, 10], dtype=dtype))

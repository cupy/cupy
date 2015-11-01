import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestCount(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    def test_count_nonzero(self, dtype):
        m = testing.shaped_random((2, 3), numpy, bool)
        a = testing.shaped_random((2, 3), numpy, dtype) * m
        c = numpy.count_nonzero(a)
        d = cupy.count_nonzero(cupy.array(a))
        self.assertEqual(d, c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_count_nonzero_zero_dim(self, xp, dtype):
        a = xp.array(1.0, dtype=dtype)
        c = xp.count_nonzero(a)
        self.assertIsInstance(c, int)
        return c

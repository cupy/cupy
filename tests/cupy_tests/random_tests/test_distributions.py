import unittest

import numpy

import cupy
from cupy.random import distributions
from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)]
})
)
@testing.gpu
class TestDistributions(unittest.TestCase):

    _multiprocess_can_split_ = True

    def check_distribution(self, dist_func, dtype):
        loc = cupy.ones(self.loc_shape)
        scale = cupy.ones(self.scale_shape)
        out = dist_func(loc, scale, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    def test_normal_float32(self):
        self.check_distribution(distributions.normal, numpy.float32)

    def test_normal_float64(self):
        self.check_distribution(distributions.normal, numpy.float64)

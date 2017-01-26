import unittest

import cupy
from cupy.random import distributions
from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributions(unittest.TestCase):

    _multiprocess_can_split_ = True

    def check_distribution(self, dist_func, loc_dtype, scale_dtype, dtype):
        loc = cupy.ones(self.loc_shape, dtype=loc_dtype)
        scale = cupy.ones(self.scale_shape, dtype=scale_dtype)
        out = dist_func(loc, scale, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_normal(self, loc_dtype, scale_dtype, dtype):
        self.check_distribution(distributions.normal,
                                loc_dtype, scale_dtype, dtype)

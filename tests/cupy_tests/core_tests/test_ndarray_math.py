import unittest

import numpy

from cupy import testing


@testing.parameterize(*testing.product({
    'decimals': [-2, -1, 0, 1, 2],
}))
class TestRound(unittest.TestCase):

    shape = (20,)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_round(self, xp, dtype):
        if dtype == numpy.bool_:
            # avoid cast problem
            a = testing.shaped_random(self.shape, xp, scale=10, dtype=dtype)
            return a.round(0)
        if dtype == numpy.float16:
            # avoid accuracy problem
            a = testing.shaped_random(self.shape, xp, scale=10, dtype=dtype)
            return a.round(0)
        a = testing.shaped_random(self.shape, xp, scale=100, dtype=dtype)
        return a.round(self.decimals)

    @testing.numpy_cupy_array_equal()
    def test_round_out(self, xp):
        a = testing.shaped_random(self.shape, xp, scale=100, dtype='d')
        out = xp.empty_like(a)
        a.round(self.decimals, out)
        return out

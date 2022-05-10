import unittest

import numpy
try:
    import scipy.spatial  # NOQA
    import scipy.spatial.distance  # NOQA
    scipy_available = True
except ImportError:
    scipy_available = False
import cupyx.scipy.spatial.distance  # NOQA
try:
    import pylibraft  # NOQA
    pylibraft_available = True
except ImportError:
    pylibraft_available = False
from cupy import testing


@testing.gpu
@testing.with_requires("scipy")
@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64'],
    'rows': [5, 10, 20],
    'cols': [5, 10, 20],
    'metric': ['euclidean', 'cityblock', 'canberra', 'chebyshev',
               'hamming', 'correlation']
}))
@unittest.skipUnless(scipy_available and pylibraft_available,
                     'requires scipy and pylibcugraph')
class TestCdist(unittest.TestCase):

    def _make_matrix(self, xp, dtype):
        shape = (self.rows, self.cols)
        return testing.shaped_random(shape, xp, dtype=dtype, scale=1)

    @testing.numpy_cupy_array_almost_equal(decimal=4, scipy_name='scp')
    def test_cdist_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype)
        out = scp.spatial.distance.cdist(a, a, metric=self.metric).astype(self.dtype)

        print(str(out.shape))
        return out

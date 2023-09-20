
from cupy import testing
import cupyx.scipy.stats.mstats  # NOQA

import pytest

try:
    import scipy.stats.mstats  # NOQA
except ImportError:
    pass


class TestMquantiles:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_mquantiles_limit_keyword(self, xp, scp):
        data = xp.array([[6., 7., 1.],
                         [47., 15., 2.],
                         [49., 36., 3.],
                         [15., 39., 4.],
                         [42., 40., -999.],
                         [41., 41., -999.],
                         [7., -999., -999.],
                         [39., -999., -999.],
                         [43., -999., -999.],
                         [40., -999., -999.],
                         [36., -999., -999.]])

        quants = scp.stats.mstats.mquantiles(data, axis=0, limit=(0, 50))
        return quants


class TestPercentile:
    @pytest.mark.parametrize('percentile', [0, 100, 50])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_percentile(self, percentile, xp, scp):
        x = xp.arange(8) * 0.5
        return scp.stats.mstats.scoreatpercentile(x, percentile)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D(self, xp, scp):
        x = xp.array([[1, 1, 1],
                      [1, 1, 1],
                      [4, 4, 3],
                      [1, 1, 1],
                      [1, 1, 1]])
        return scp.stats.mstats.scoreatpercentile(x, 50)

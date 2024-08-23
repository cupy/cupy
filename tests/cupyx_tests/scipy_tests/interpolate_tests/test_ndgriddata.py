
import cupy
from cupy import testing

import cupyx.scipy.interpolate  # NOQA

try:
    import scipy.interpolate  # NOQA
except ImportError:
    pass

import numpy as np
import pytest


class TestNearestNDInterpolator:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nearest_options(self, xp, scp):
        # smoke test that NearestNDInterpolator accept cKDTree options
        npts, nd = 4, 3
        x = xp.arange(npts * nd).reshape((npts, nd))
        y = xp.arange(npts)
        nndi = scp.interpolate.NearestNDInterpolator(x, y)
        return nndi(x).astype(xp.float64)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nearest_list_argument(self, xp, scp):
        nd = xp.array([[0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1, 2]])
        d = nd[:, 3:]

        # z is np.array
        NI = scp.interpolate.NearestNDInterpolator((d[0], d[1]), d[2])
        return NI(xp.asarray([0.1, 0.9]),
                  xp.asarray([0.1, 0.9])).astype(xp.float64)

    @testing.with_requires('scipy>=1.12')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nearest_query_options(self, xp, scp):
        nd = xp.array([[0, 0.5, 0, 1],
                       [0, 0, 0.5, 1],
                       [0, 1, 1, 2]])
        delta = 0.1
        query_points = (xp.asarray([0 + delta, 1 + delta]),
                        xp.asarray([0 + delta, 1 + delta]))

        # case 1 - query max_dist is smaller than
        # the query points' nearest distance to nd.
        NI = scp.interpolate.NearestNDInterpolator((nd[0], nd[1]), nd[2])
        distance_upper_bound = (np.sqrt(delta ** 2 + delta ** 2) - 1e-7).item()
        r1 = NI(query_points, distance_upper_bound=distance_upper_bound)

        # case 2 - query p is inf, will return [0, 2]
        distance_upper_bound = (np.sqrt(delta ** 2 + delta ** 2) - 1e-7).item()
        p = xp.inf
        r2 = NI(query_points, distance_upper_bound=distance_upper_bound, p=p)

        # case 3 - query max_dist is larger, so should return non np.nan
        distance_upper_bound = (np.sqrt(delta ** 2 + delta ** 2) + 1e-7).item()
        r3 = NI(query_points, distance_upper_bound=distance_upper_bound)
        return r1, r2, r3

    @testing.with_requires('scipy>=1.12')
    @pytest.mark.parametrize('xp,scp', [(np, scipy), (cupy, cupyx.scipy)])
    def test_nearest_query_valid_inputs(self, xp, scp):
        nd = xp.array([[0, 1, 0, 1],
                       [0, 0, 1, 1],
                       [0, 1, 1, 2]])
        NI = scp.interpolate.NearestNDInterpolator((nd[0], nd[1]), nd[2])
        with pytest.raises(TypeError):
            NI([0.5, 0.5], query_options="not a dictionary")

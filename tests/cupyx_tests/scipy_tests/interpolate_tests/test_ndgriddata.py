from __future__ import annotations


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


@testing.with_requires("scipy")
class TestGriddata:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fill_value(self, xp, scp):
        x = [(0, 0), (0, 1), (1, 0)]
        y = [1, 2, 3]

        yi = scp.interpolate.griddata(x, y, [(1, 1), (1, 2), (0, 0)],
                                      fill_value=-1)
        yi_ = scp.interpolate.griddata(x, y, [(1, 1), (1, 2), (0, 0)])
        return yi, yi_

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @pytest.mark.parametrize('rescale', (True, False))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=5e-16)
    def test_alternative_call(self, xp, scp, method, rescale):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5),
                     (0.25, 0.3)], dtype=xp.float64)
        y = (xp.arange(x.shape[0], dtype=xp.float64)[:, None]
             + xp.array([0, 1])[None, :])

        yi = scp.interpolate.griddata((x[:, 0], x[:, 1]), y,
                                      (x[:, 0], x[:, 1]),
                                      method=method, rescale=rescale)
        return yi

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @pytest.mark.parametrize('rescale', (True, False))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=5e-16)
    def test_multivalue_2d(self, xp, scp, method, rescale):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5),
                      (0.25, 0.3)], dtype=xp.float64)
        y = (xp.arange(x.shape[0], dtype=xp.float64)[:, None]
             + xp.array([0, 1])[None, :])

        yi = scp.interpolate.griddata(x, y, x, method=method, rescale=rescale)
        return yi

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @pytest.mark.parametrize('rescale', (True, False))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=5e-16)
    def test_multipoint_2d(self, xp, scp, method, rescale):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5),
                      (0.25, 0.3)], dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        xi = x[:, None, :] + xp.array([0, 0, 0])[None, :, None]
        yi = scp.interpolate.griddata(x, y, xi, method=method, rescale=rescale)
        return yi

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @pytest.mark.parametrize('rescale', (True, False))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_2d(self, xp, scp, method, rescale):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5),
                      (0.25, 0.3)], dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 2j*y[::-1]

        xi = x[:, None, :] + xp.array([0, 0, 0])[None, :, None]

        yi = scp.interpolate.griddata(x, y, xi, method=method, rescale=rescale)
        return yi

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-15)
    def test_1d(self, xp, scp, method):
        x = xp.array([1, 2.5, 3, 4.5, 5, 6])
        y = xp.array([1, 2, 0, 3.9, 2, 1])

        return (scp.interpolate.griddata(x, y, x, method=method),
                scp.interpolate.griddata(x.reshape(6, 1), y, x, method=method),
                scp.interpolate.griddata((x,), y, (x,), method=method)
                )

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_1d_borders(self, xp, scp):
        # Test for nearest neighbor case with xi outside
        # the range of the values.
        x = xp.array([1, 2.5, 3, 4.5, 5, 6])
        y = xp.array([1, 2, 0, 3.9, 2, 1])
        xi = xp.array([0.9, 6.5])

        method = 'nearest'
        return (scp.interpolate.griddata(x, y, xi, method=method),
                scp.interpolate.griddata(
                    x.reshape(6, 1), y, xi, method=method),
                scp.interpolate.griddata((x, ), y, (xi, ), method=method)
                )

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-15)
    def test_1d_unsorted(self, xp, scp, method):
        x = xp.array([2.5, 1, 4.5, 5, 6, 3])
        y = xp.array([1, 2, 0, 3.9, 2, 1])

        return (scp.interpolate.griddata(x, y, x, method=method),
                scp.interpolate.griddata(x.reshape(6, 1), y, x, method=method),
                scp.interpolate.griddata((x,), y, (x,), method=method)
                )

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5)
    def test_square_rescale_manual(self, xp, scp, method):
        points = xp.array([(0, 0), (0, 100), (10, 100), (10, 0), (1, 5)],
                          dtype=xp.float64)
        points_rescaled = xp.array([(0, 0), (0, 1), (1, 1), (1, 0),
                                   (0.1, 0.05)], dtype=xp.float64)
        values = xp.array([1., 2., -3., 5., 9.], dtype=xp.float64)

        xx, yy = xp.broadcast_arrays(xp.linspace(0, 10, 14)[:, None],
                                     xp.linspace(0, 100, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = xp.array([xx, yy]).T.copy()

        zi = scp.interpolate.griddata(points_rescaled, values,
                                      xi/xp.array([10, 100.]), method=method)
        zi_rescaled = scp.interpolate.griddata(points, values, xi,
                                               method=method, rescale=True)
        return zi, zi_rescaled

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-15)
    def test_xi_1d(self, xp, scp, method):
        # Check that 1-D xi is interpreted as a coordinate
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5),
                      (0.25, 0.3)], dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 2j*y[::-1]
        xi = xp.array([0.5, 0.5])
        return (scp.interpolate.griddata(x, y, xi, method=method),
                scp.interpolate.griddata(x, y, xi[None, :], method=method))

    @pytest.mark.parametrize('method', ('nearest', 'linear', 'cubic'))
    def test_xi_1d_raises(self, method):
        x = cupy.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5),
                        (0.25, 0.3)], dtype=cupy.float64)
        y = cupy.arange(x.shape[0], dtype=cupy.float64)
        y = y - 2j*y[::-1]

        xi1 = cupy.array([0.5])
        xi3 = cupy.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError):
            cupyx.scipy.interpolate.griddata(x, y, xi1, method=method)

        with pytest.raises(ValueError):
            cupyx.scipy.interpolate.griddata(x, y, xi3, method=method)


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


class TestGriddata:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fill_value(self, xp, scp):
        x = [(0, 0), (0, 1), (1, 0)]
        y = [1, 2, 3]

        yi1 = scp.interpolate.griddata(
            x, y, [(1, 1), (1, 2), (0, 0)], fill_value=-1)
        expected1 = xp.array([-1., -1, 1], dtype=xp.float64)

        yi2 = scp.interpolate.griddata(x, y, [(1, 1), (1, 2), (0, 0)])
        expected2 = xp.array([xp.nan, xp.nan, 1], dtype=xp.float64)

        return xp.stack([yi1, yi2]), xp.stack([expected1, expected2])

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @pytest.mark.parametrize("rescale", [True, False])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_alternative_call(self, method, rescale, xp, scp):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)], dtype=xp.float64)
        y = (xp.arange(x.shape[0], dtype=xp.float64)[:, None]
             + xp.array([0, 1], dtype=xp.float64)[None, :])

        yi = scp.interpolate.griddata((x[:, 0], x[:, 1]), y,
                                      (x[:, 0], x[:, 1]),
                                      method=method, rescale=rescale)

        return yi, y

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @pytest.mark.parametrize("rescale", [True, False])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_multivalue_2d(self, method, rescale, xp, scp):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)], dtype=xp.float64)
        y = (xp.arange(x.shape[0], dtype=xp.float64)[:, None]
             + xp.array([0, 1], dtype=xp.float64)[None, :])

        yi = scp.interpolate.griddata(x, y, x, method=method, rescale=rescale)

        return yi, y

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @pytest.mark.parametrize("rescale", [True, False])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_multipoint_2d(self, method, rescale, xp, scp):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)], dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)

        xi = x[:, None, :] + \
            xp.array([0, 0, 0], dtype=xp.float64)[None, :, None]

        yi = scp.interpolate.griddata(x, y, xi, method=method, rescale=rescale)
        expected = xp.tile(y[:, None], (1, 3))

        return yi, expected

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @pytest.mark.parametrize("rescale", [True, False])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_complex_2d(self, method, rescale, xp, scp):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)], dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.complex128)
        y = y - 2j * y[::-1]

        xi = x[:, None, :] + \
            xp.array([0, 0, 0], dtype=xp.float64)[None, :, None]

        yi = scp.interpolate.griddata(x, y, xi, method=method, rescale=rescale)

        expected = xp.tile(y[:, None], (1, 3))

        return yi, expected

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_1d(self, method, xp, scp):
        x = xp.array([1, 2.5, 3, 4.5, 5, 6])
        y = xp.array([1, 2, 0, 3.9, 2, 1])

        yi1 = scp.interpolate.griddata(x, y, x, method=method)
        yi2 = scp.interpolate.griddata(x.reshape(6, 1), y, x, method=method)
        yi3 = scp.interpolate.griddata((x,), y, (x,), method=method)

        result = xp.stack([yi1, yi2, yi3])
        expected = xp.stack([y, y, y])

        return result, expected

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_1d_borders(self, xp, scp):
        x = xp.array([1, 2.5, 3, 4.5, 5, 6])
        y = xp.array([1, 2, 0, 3.9, 2, 1])
        xi = xp.array([0.9, 6.5])
        yi_should = xp.array([1.0, 1.0])
        method = 'nearest'

        try:
            yi1 = scp.interpolate.griddata(x, y, xi, method=method)
            yi2 = scp.interpolate.griddata(
                x.reshape(6, 1), y, xi, method=method)
            yi3 = scp.interpolate.griddata((x,), y, (xi,), method=method)
        except Exception as e:
            raise AssertionError(
                f"Interpolation failed in method '{method}': {e}") from e

        return xp.stack([yi1, yi2, yi3]), xp.stack([yi_should]*3)

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-10)
    def test_1d_unsorted(self, method, xp, scp):
        x = xp.array([2.5, 1, 4.5, 5, 6, 3])
        y = xp.array([1, 2, 0, 3.9, 2, 1])

        r1 = scp.interpolate.griddata(x, y, x, method=method)
        r2 = scp.interpolate.griddata(x.reshape(6, 1), y, x, method=method)
        r3 = scp.interpolate.griddata((x,), y, (x,), method=method)
        return r1, y, r2, y, r3, y

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-6, atol=5e-7)
    def test_square_rescale_manual(self, method, xp, scp):
        points = xp.array([(0, 0), (0, 100), (10, 100),
                          (10, 0), (1, 5)], dtype=xp.float64)
        points_rescaled = xp.array(
            [(0, 0), (0, 1), (1, 1), (1, 0), (0.1, 0.05)], dtype=xp.float64)
        values = xp.array([1., 2., -3., 5., 9.], dtype=xp.float64)

        xx, yy = xp.broadcast_arrays(xp.linspace(
            0, 10, 14)[:, None], xp.linspace(0, 100, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = xp.array([xx, yy]).T.copy()

        zi = scp.interpolate.griddata(
            points_rescaled, values, xi / xp.array([10, 100.]), method=method)
        zi_rescaled = scp.interpolate.griddata(
            points, values, xi, method=method, rescale=True)
        return zi, zi_rescaled

    @pytest.mark.parametrize("method", ['linear', 'nearest', 'cubic'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_xi_1d(self, method, xp, scp):
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                     (0.5, 0.5), (0.25, 0.3)], dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 2j*y[::-1]

        xi = xp.array([0.5, 0.5])
        p1 = scp.interpolate.griddata(x, y, xi, method=method)
        p2 = scp.interpolate.griddata(x, y, xi[None, :], method=method)

        return p1, p2

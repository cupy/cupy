
from cupy import testing

import cupy  # NOQA
import cupyx.scipy.spatial  # NOQA

import pytest
import numpy as np

try:
    import scipy.spatial  # NOQA
except ImportError:
    pass


pathological_data_1 = np.array([
    [-3.14, -3.14], [-3.14, -2.36], [-3.14, -1.57], [-3.14, -0.79],
    [-3.14, 0.0], [-3.14, 0.79], [-3.14, 1.57], [-3.14, 2.36],
    [-3.14, 3.14], [-2.36, -3.14], [-2.36, -2.36], [-2.36, -1.57],
    [-2.36, -0.79], [-2.36, 0.0], [-2.36, 0.79], [-2.36, 1.57],
    [-2.36, 2.36], [-2.36, 3.14], [-1.57, -0.79], [-1.57, 0.79],
    [-1.57, -1.57], [-1.57, 0.0], [-1.57, 1.57], [-1.57, -3.14],
    [-1.57, -2.36], [-1.57, 2.36], [-1.57, 3.14], [-0.79, -1.57],
    [-0.79, 1.57], [-0.79, -3.14], [-0.79, -2.36], [-0.79, -0.79],
    [-0.79, 0.0], [-0.79, 0.79], [-0.79, 2.36], [-0.79, 3.14],
    [0.0, -3.14], [0.0, -2.36], [0.0, -1.57], [0.0, -0.79], [0.0, 0.0],
    [0.0, 0.79], [0.0, 1.57], [0.0, 2.36], [0.0, 3.14], [0.79, -3.14],
    [0.79, -2.36], [0.79, -0.79], [0.79, 0.0], [0.79, 0.79],
    [0.79, 2.36], [0.79, 3.14], [0.79, -1.57], [0.79, 1.57],
    [1.57, -3.14], [1.57, -2.36], [1.57, 2.36], [1.57, 3.14],
    [1.57, -1.57], [1.57, 0.0], [1.57, 1.57], [1.57, -0.79],
    [1.57, 0.79], [2.36, -3.14], [2.36, -2.36], [2.36, -1.57],
    [2.36, -0.79], [2.36, 0.0], [2.36, 0.79], [2.36, 1.57],
    [2.36, 2.36], [2.36, 3.14], [3.14, -3.14], [3.14, -2.36],
    [3.14, -1.57], [3.14, -0.79], [3.14, 0.0], [3.14, 0.79],
    [3.14, 1.57], [3.14, 2.36], [3.14, 3.14],
])

pathological_data_2 = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 0], [0, 1],
    [1, -1 - np.finfo(np.float64).eps], [1, 0], [1, 1],
])


class TestDelaunay:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2d_triangulation(self, xp, scp):
        points = testing.shaped_random((100, 2), xp, xp.float64)
        tri = scp.spatial.Delaunay(points)
        return xp.sort(xp.sort(tri.simplices, axis=-1), axis=0)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2d_square(self, xp, scp):
        # simple smoke test: 2d square
        points = xp.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=xp.float64)
        tri = scp.spatial.Delaunay(points)
        return xp.sort(xp.sort(tri.simplices, axis=-1), axis=0)

    @pytest.mark.parametrize(
        'dataset', [pathological_data_1, pathological_data_2])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pathological(self, dataset, xp, scp):
        points = xp.asarray(dataset, dtype=xp.float64)
        tri = scp.spatial.Delaunay(points)
        return points[tri.simplices].max(), points[tri.simplices].min()

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_duplicate_points(self, mod):
        xp, scp = mod
        x = xp.array([0, 1, 0, 1], dtype=xp.float64)
        y = xp.array([0, 0, 1, 1], dtype=xp.float64)

        x_p = xp.r_[x, x]
        y_p = xp.r_[y, y]

        scp.spatial.Delaunay(xp.c_[x, y])
        scp.spatial.Delaunay(xp.c_[x_p, y_p])

    @pytest.mark.parametrize(
        'point', [(0.25, 0.5), (0.5, 0.25), (0.75, 0.5), (0.5, 0.75)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_find_simplex(self, point, xp, scp):
        points = xp.array([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5)],
                          dtype=xp.float64)
        tri = scp.spatial.Delaunay(points)

        search_point = xp.asarray([point], dtype=xp.float64)
        tri_index = tri.find_simplex(search_point)
        triangle = tri.simplices[tri_index]
        return xp.sort(triangle)

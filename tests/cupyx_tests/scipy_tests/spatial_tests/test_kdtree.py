
import pytest

import cupy
from cupy import testing
import cupyx.scipy.spatial  # NOQA

import numpy as np

try:
    import scipy  # NOQA
except ImportError:
    pass

try:
    import scipy.spatial  # NOQA
except ImportError:
    pass


def create_random_kd_tree(xp, scp, n, m, n_points=1, x_offset=0, scale=10,
                          leafsize=100):
    data = testing.shaped_random((n, m), xp, xp.float64, scale=scale,
                                 seed=1234)
    x = testing.shaped_random((n_points, m), xp, xp.float64, scale) + x_offset
    tree = scp.spatial.KDTree(data, leafsize=leafsize)
    return x, tree


def create_small_kd_tree(xp, scp, n_points=1):
    data = xp.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0],
                     [1, 1, 1.]])
    tree = scp.spatial.KDTree(data)
    _, m = data.shape
    x = testing.shaped_random((n_points, m), xp, xp.float64)
    return x, tree


@testing.with_requires('scipy')
class TestRandomConsistency:
    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nearest(self, xp, scp, args):
        x, tree = create_random_kd_tree(xp, scp, *args)
        d, i = tree.query(x, 1)
        return d, i

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_m_nearest(self, xp, scp, args):
        m = args[1]
        x, tree = create_random_kd_tree(xp, scp, *args)
        dd, ii = tree.query(x, m)
        return dd, ii

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_points_near(self, xp, scp, args):
        n = args[0]
        x, tree = create_random_kd_tree(xp, scp, *args)
        d = 0.2
        dd, ii = tree.query(x, k=n, distance_upper_bound=d)
        return dd, ii

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_points_near_l1(self, xp, scp, args):
        n = args[0]
        x, tree = create_random_kd_tree(xp, scp, *args)
        d = 0.2
        dd, ii = tree.query(x, k=n, p=1, distance_upper_bound=d)
        return dd, ii

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_points_near_linf(self, xp, scp, args):
        n = args[0]
        x, tree = create_random_kd_tree(xp, scp, *args)
        d = 0.2
        dd, ii = tree.query(x, k=n, p=xp.inf, distance_upper_bound=d)
        return dd, ii

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_approx(self, xp, scp, args):
        k = 10
        eps = 0.1
        x, tree = create_random_kd_tree(xp, scp, *args)
        d, i = tree.query(x, k, eps=eps)
        return d, i

    @pytest.mark.parametrize('k', [(1, 2, 3), (1, 3), (1,)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_list_k(self, xp, scp, k):
        n = 200
        m = 2
        data = testing.shaped_random(
            (n, m), xp, xp.float64, scale=1, seed=1234)
        kdtree = scp.spatial.KDTree(data)
        dd, ii = kdtree.query(data, list(k))
        return dd, ii


@testing.with_requires('scipy')
class TestSmall:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nearest(self, xp, scp):
        x, tree = create_small_kd_tree(xp, scp)
        d, i = tree.query(x, 1)
        return d, i

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_m_nearest(self, xp, scp):
        x, tree = create_small_kd_tree(xp, scp)
        m = tree.m
        dd, ii = tree.query(x, m)
        return dd, ii

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_points_near(self, xp, scp):
        x, tree = create_small_kd_tree(xp, scp)
        n = tree.n
        d = 0.2
        dd, ii = tree.query(x, k=n, distance_upper_bound=d)
        return dd, ii

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_points_near_l1(self, xp, scp):
        x, tree = create_small_kd_tree(xp, scp)
        n = tree.n
        d = 0.2
        dd, ii = tree.query(x, k=n, p=1, distance_upper_bound=d)
        return dd, ii

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_points_near_linf(self, xp, scp):
        x, tree = create_small_kd_tree(xp, scp)
        n = tree.n
        d = 0.2
        dd, ii = tree.query(x, k=n, p=xp.inf, distance_upper_bound=d)
        return dd, ii

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_approx(self, xp, scp):
        k = 10
        eps = 0.1
        x, tree = create_small_kd_tree(xp, scp)
        d, i = tree.query(x, k, eps=eps)
        return d, i


@testing.with_requires('scipy')
class TestVectorization:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vectorized_query(self, xp, scp):
        _, tree = create_small_kd_tree(xp, scp)
        d, i = tree.query(xp.zeros((2, 4, 3)))
        return d, i

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_single_query_multiple_neighbors(self, xp, scp):
        s = 23
        _, tree = create_small_kd_tree(xp, scp)
        kk = tree.n + s
        d, _ = tree.query(xp.array([0, 0, 0]), k=kk)
        return d

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vectorized_query_multiple_neighbors(self, xp, scp):
        s = 23
        _, tree = create_small_kd_tree(xp, scp)
        kk = tree.n + s
        d, _ = tree.query(xp.zeros((2, 4, 3)), k=kk)
        return d

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vectorized_query_multiple_neighbors_random(self, xp, scp):
        s = 23
        x, tree = create_random_kd_tree(xp, scp, 100, 3, 8)
        x = x.reshape(2, 4, 3).copy()
        kk = tree.n + s
        d, _ = tree.query(x, k=kk)
        return d

    @testing.with_requires('scipy>=1.9.0')
    def test_query_raises_for_k_none(self):
        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            x, tree = create_small_kd_tree(xp, scp)
            with pytest.raises(ValueError, match="k must be an integer*"):
                tree.query(x, k=None)


class TestPeriodic:
    @pytest.mark.parametrize('off', [0, 1, -1])
    @pytest.mark.parametrize('p', [1, 2, 3.0, np.inf])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kdtree_box(self, xp, scp, p, off):
        # check ckdtree periodic boundary
        n = 100
        m = 2
        k = 3

        data = testing.shaped_random(
            (n, m), xp, xp.float64, scale=1, seed=1234)
        kdtree = scp.spatial.KDTree(data, boxsize=1.0)

        dd, ii = kdtree.query(data + off, k, p=p)
        return dd, ii

    @pytest.mark.parametrize('off', [0, 1, -1])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kdtree_ball(self, xp, scp, off):
        # check ckdtree periodic boundary
        n = 100
        m = 2

        data = testing.shaped_random(
            (n, m), xp, xp.float64, scale=1, seed=1234)
        kdtree = scp.spatial.KDTree(data, boxsize=1.0, leafsize=100)

        res = kdtree.query_ball_point(data + off, 0.5, return_sorted=True)
        if xp is not cupy:
            res = [xp.asarray(r, dtype=xp.int64) for r in res]
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kdtree_ball_tree(self, xp, scp):
        # check ckdtree periodic boundary
        n = 100
        m = 2

        data1 = testing.shaped_random(
            (n, m), xp, xp.float64, scale=1, seed=1234)
        kdtree = scp.spatial.KDTree(data1, boxsize=1.0, leafsize=100)

        data2 = testing.shaped_random(
            (n, m), xp, xp.float64, scale=1, seed=1234)
        kdtree2 = scp.spatial.KDTree(data2, boxsize=1.0, leafsize=100)

        res = kdtree.query_ball_tree(kdtree2, 0.5)
        if xp is not cupy:
            res = [xp.asarray(r, dtype=xp.int64) for r in res]
        return res


@testing.with_requires('scipy')
class TestBallConsistency:
    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_in_ball(self, xp, scp, args):
        x, tree = create_random_kd_tree(xp, scp, *args, scale=1.0)
        res = tree.query_ball_point(
            x, 0.5, return_sorted=True)
        if xp is not cupy:
            res = [xp.asarray(r) for r in res]
        return res

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_approx(self, xp, scp, args):
        eps = 0.1
        x, tree = create_random_kd_tree(xp, scp, *args, scale=1.0)
        res = tree.query_ball_point(x, 0.5, return_sorted=True, eps=eps)
        if xp is not cupy:
            res = [xp.asarray(r) for r in res]
        return res


@testing.with_requires('scipy')
class TestBallTreeConsistency:
    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_in_ball(self, xp, scp, args):
        _, tree = create_random_kd_tree(xp, scp, *args, scale=1.0)
        _, tree2 = create_random_kd_tree(xp, scp, *args, scale=1.0)
        res = tree.query_ball_tree(tree2, 0.5)
        if xp is not cupy:
            res = [xp.asarray(r) for r in res]
        return res

    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_approx(self, xp, scp, args):
        eps = 0.1
        _, tree = create_random_kd_tree(xp, scp, *args, scale=1.0)
        _, tree2 = create_random_kd_tree(xp, scp, *args, scale=1.0)
        res = tree.query_ball_tree(tree2, 0.5, eps=eps)
        if xp is not cupy:
            res = [xp.asarray(r) for r in res]
        return res


@testing.with_requires('scipy')
class TestPairs:
    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', type_check=False)
    def test_find_pairs(self, xp, scp, args):
        _, tree = create_random_kd_tree(xp, scp, *args, scale=1.0)
        res = tree.query_pairs(0.5, output_type='ndarray')
        return res

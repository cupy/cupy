
import pytest

from cupy import testing
import cupyx.scipy.spatial  # NOQA

try:
    import scipy.spatial  # NOQA
except ImportError:
    pass


def create_random_kd_tree(xp, scp, n, m, n_points=1, x_offset=0):
    data = testing.shaped_random((n, m), xp, xp.float64, seed=1234)
    x = testing.shaped_random((n_points, m), xp, xp.float64) + x_offset
    tree = scp.spatial.KDTree(data)
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

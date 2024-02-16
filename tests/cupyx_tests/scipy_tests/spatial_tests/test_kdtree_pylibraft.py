import pytest

from cupy import testing

try:
    import pylibraft  # NOQA
    pylibraft_available = True
except ModuleNotFoundError:
    pylibraft_available = False


def create_random_kd_tree(xp, scp, n, m, n_points=1, x_offset=0, scale=10,
                          leafsize=100):
    data = testing.shaped_random((n, m), xp, xp.float64, scale=scale,
                                 seed=1234)
    x = testing.shaped_random((n_points, m), xp, xp.float64, scale) + x_offset
    tree = scp.spatial.KDTree(data, leafsize=leafsize)
    return x, tree


@testing.with_requires('scipy')
@pytest.mark.skipif(not pylibraft_available, reason='pylibraft is required')
class TestDistance:
    @pytest.mark.parametrize('args', [
        (100, 4, 1, 0),
        (100, 4, 1, 10)
    ])
    @testing.numpy_cupy_allclose(
        scipy_name='scp', type_check=False, rtol=5e-7, atol=5e-7)
    def test_sparse_distance_matrix(self, xp, scp, args):
        _, tree = create_random_kd_tree(xp, scp, *args, scale=1.0)
        _, tree2 = create_random_kd_tree(xp, scp, *args, scale=1.0)
        res = tree.sparse_distance_matrix(tree2, 0.5, output_type='coo_matrix')
        return res.todense()

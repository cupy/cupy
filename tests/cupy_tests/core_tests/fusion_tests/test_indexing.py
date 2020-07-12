import unittest

from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


@testing.parameterize(
    # Integer
    {'shape': (2, 3, 4), 'indices': 1},
    {'shape': (2, 3, 4), 'indices': -1},
    {'shape': (2, 3, 4), 'indices': (1,)},
    {'shape': (2, 3, 4), 'indices': (1, 0)},
    {'shape': (2, 3, 4), 'indices': (1, 0, 2)},
    {'shape': (2, 3, 4), 'indices': (-1, 0, -2)},
    # Slice
    {'shape': (2, 3, 4), 'indices': slice(None)},
    {'shape': (2, 3, 4), 'indices': slice(None, None, 1)},
    {'shape': (2, 3, 4), 'indices': slice(None, None, -1)},
    {'shape': (2, 3, 4), 'indices': (slice(None), slice(None, None, -1))},
    # Ellipsis
    {'shape': (2, 3, 4), 'indices': Ellipsis},
    {'shape': (2, 3, 4), 'indices': (Ellipsis,)},
    # Newaxis
    {'shape': (2, 3, 4), 'indices': None},
    {'shape': (2, 3, 4), 'indices': (None,)},
    {'shape': (2, 3, 4), 'indices': (None, None, None)},
    # Tuple
    {'shape': (2, 3, 4), 'indices': (slice(None), 0, slice(None, None, -1))},
    {'shape': (2, 3, 4), 'indices': (1, None, slice(None, None, -1), None, 2)},
    {'shape': (2,), 'indices': (slice(None), None)},
    {'shape': (2, 3, 4), 'indices': (Ellipsis, 2)},
    {'shape': (2, 3, 4), 'indices': (1, Ellipsis)},
    {'shape': (2, 3, 4, 5), 'indices': (1, Ellipsis, 3)},
)
@testing.gpu
class TestIndexing(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype, scale=10, seed=0)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_getitem(self, xp, dtype):
        return lambda x: x[self.indices]


@testing.parameterize(
    {'shape': (), 'indices': 0},
    {'shape': (2, 3), 'indices': (0, 0, 0)},
    {'shape': (2, 3, 4), 'indices': -3},
    {'shape': (2, 3, 4), 'indices': 3},
)
@testing.with_requires('numpy>=1.12.0')
@testing.gpu
class TestArrayInvalidIndex(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(accept_error=(IndexError,))
    def test_invalid_getitem(self, xp, dtype):
        return lambda x: x[self.indices]


@testing.gpu
class TestIndexingCombination(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((4,), xp, dtype2, scale=10, seed=1)
        z = testing.shaped_random((1,), xp, dtype1, scale=10, seed=2)
        return (x, y, z), {}

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_and_add_1(self, xp, dtype1, dtype2):
        return lambda x, y, z: x + y[1]

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_and_add_2(self, xp, dtype1, dtype2):
        return lambda x, y, z: x + z[0] + y

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_and_add_3(self, xp, dtype1, dtype2):
        return lambda x, y, z: x + x[0] + x[1]

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_and_add_4(self, xp, dtype1, dtype2):
        return lambda x, y, z: x + x[0, 1] + x[1] + x + x[2, 1]

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_twice_1(self, xp, dtype1, dtype2):
        return lambda x, y, z: x[0][1]

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_twice_2(self, xp, dtype1, dtype2):
        return lambda x, y, z: x[0][1] + x[1][0]

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_twice_3(self, xp, dtype1, dtype2):
        return lambda x, y, z: x[0][1] + x[1] + y[0] + x[1][0] + x

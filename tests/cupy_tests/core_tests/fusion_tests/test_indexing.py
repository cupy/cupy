from __future__ import annotations

import pytest

from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


@pytest.mark.parametrize(
    "shape,indices",
    [
        # Integer
        ((2, 3, 4), 1),
        ((2, 3, 4), -1),
        ((2, 3, 4), (1,)),
        ((2, 3, 4), (1, 0)),
        ((2, 3, 4), (1, 0, 2)),
        ((2, 3, 4), (-1, 0, -2)),
        # Slice
        ((2, 3, 4), slice(None)),
        ((2, 3, 4), slice(None, None, 1)),
        ((2, 3, 4), slice(None, None, -1)),
        ((2, 3, 4), (slice(None), slice(None, None, -1))),
        # Ellipsis
        ((2, 3, 4), Ellipsis),
        ((2, 3, 4), (Ellipsis,)),
        # Newaxis
        ((2, 3, 4), None),
        ((2, 3, 4), (None,)),
        ((2, 3, 4), (None, None, None)),
        # Tuple
        ((2, 3, 4), (slice(None), 0, slice(None, None, -1))),
        ((2, 3, 4), (1, None, slice(None, None, -1), None, 2)),
        ((2,), (slice(None), None)),
        ((2, 3, 4), (Ellipsis, 2)),
        ((2, 3, 4), (1, Ellipsis)),
        ((2, 3, 4, 5), (1, Ellipsis, 3)),
    ]
)
class TestIndexing:

    def generate_inputs(self, xp, dtype, shape):
        x = testing.shaped_random(shape, xp, dtype, scale=10, seed=0)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_getitem(self, xp, dtype, shape, indices):
        return lambda x: x[indices]


@pytest.mark.parametrize(
    "shape,indices",
    [
        ((), 0),
        ((2, 3), (0, 0, 0)),
        ((2, 3, 4), -3),
        ((2, 3, 4), 3),
    ]
)
@testing.with_requires('numpy>=1.12.0')
class TestArrayInvalidIndex:

    def generate_inputs(self, xp, dtype, shape):
        x = testing.shaped_arange(shape, xp, dtype)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(accept_error=(IndexError,))
    def test_invalid_getitem(self, xp, dtype, shape, indices):
        return lambda x: x[indices]


class TestIndexingCombination:

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

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @fusion_utils.check_fusion()
    def test_indexing_twice_3(self, xp, dtype1, dtype2):
        return lambda x, y, z: x[0][1] + x[1] + y[0] + x[1][0] + x

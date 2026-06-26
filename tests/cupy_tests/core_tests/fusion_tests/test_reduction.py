from __future__ import annotations

import cupy
import pytest

from cupy import testing
from cupy.exceptions import AxisError
from cupy_tests.core_tests.fusion_tests import fusion_utils


@pytest.mark.parametrize(
    "shape",
    [(1,), (3, 4), (2, 1, 4), (2, 0, 3)]
)
@pytest.mark.parametrize(
    "axis", [-4, -3, -2, -1, 0, 1, 2, 3, 4]
)
class TestFusionReductionAxis:

    def generate_inputs(self, xp, shape):
        x = testing.shaped_random(shape, xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion(accept_error=AxisError)
    def test_sum_axis(self, xp, shape, axis):
        return lambda x: cupy.sum(x, axis)

    @fusion_utils.check_fusion(accept_error=AxisError)
    def test_sum_kwargs_axis(self, xp, shape, axis):
        return lambda x: cupy.sum(x, axis=axis)


@pytest.mark.parametrize(
    "shape",
    [(1,), (3, 4), (2, 1, 4), (2, 0, 3), (2, 3, 2, 2, 3)]
)
@pytest.mark.parametrize(
    "axis",
    [
        None, (0,), (1,), (0, 1), (1, 2), (0, 2), (1, 3)
        # TODO(asi1024): Fix core.simple_reduction_kernel to raise Error.
        # (0, 0), (-1, 1)
    ]
)
class TestFusionReductionMultiAxis:

    def generate_inputs(self, xp, shape):
        x = testing.shaped_random(shape, xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion(accept_error=(ValueError, AxisError))
    def test_sum_axis(self, xp, shape, axis):
        return lambda x: cupy.sum(x, axis)

    @fusion_utils.check_fusion(accept_error=(ValueError, AxisError))
    def test_sum_kwargs_axis(self, xp, shape, axis):
        return lambda x: cupy.sum(x, axis=axis)


@pytest.mark.parametrize(
    "shape",
    [
        (120, 128, 144),
        (119, 127, 143),
        (128, 128, 128),
        (32, 1024, 1024)
    ]
)
@pytest.mark.parametrize(
    "axis",
    [None, 0, 1, 2, (0, 1), (0, 2), (1, 2)],
)
@testing.slow
class TestFusionReductionLarge:

    def generate_inputs(self, xp, shape):
        x = testing.shaped_random(shape, xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion()
    def test_sum_kwargs_axis(self, xp, shape, axis):
        return lambda x: cupy.sum(x, axis=axis)


# TODO(asi1024): Support for bool and complex dtypes.
class TestFusionReductionSpecifyDtype:

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return (x,), {}

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True, no_complex=True)
    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_sum(self, xp, dtype1, dtype2):
        return lambda x: x.sum(axis=0, dtype=dtype2)


@pytest.mark.parametrize("axis", [None, 0, 1])
class TestFusionReductionAndElementwise:

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=1)
        return (x, y), {}

    @fusion_utils.check_fusion()
    def test_premap_one_array(self, xp, axis):
        return lambda x, y: xp.sum(x * 3, axis)

    @fusion_utils.check_fusion()
    def test_premap_two_arrays(self, xp, axis):
        return lambda x, y: xp.sum(x + y, axis)

    @fusion_utils.check_fusion()
    def test_postmap_one_array(self, xp, axis):
        return lambda x, y: xp.sum(x, axis) + 3

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_postmap_two_arrays(self, xp, axis):
        return lambda x, y: xp.sum(x, axis) + y

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_premap_postmap(self, xp, axis):
        return lambda x, y: xp.sum(xp.sqrt(x) + y, axis) * 2 + y

    # TODO(asi1024): Uncomment after replace fusion implementation.
    # @fusion_utils.check_fusion()
    # def test_premap_inplace(self, xp):
    #     def impl(x, y):
    #         x += 2
    #         y += x
    #         return xp.sum(y, self.axis)
    #     return impl

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_postmap_inplace(self, xp, axis):
        def impl(x, y):
            y += x
            res = xp.sum(x, axis)
            y += res
        return impl


@pytest.mark.parametrize("axis1", [None, 0, 1])
@pytest.mark.parametrize("axis2", [None, 0, 1])
class TestFusionMultipleReductions:

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=1)
        return (x, y), {}

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_two_distinct_reductions(self, xp, axis1, axis2):
        return lambda x, y: (x.sum(axis1), y.sum(axis2))

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_two_reductions_and_elementwise(self, xp, axis1, axis2):
        return lambda x, y: x.sum(axis1) + y.sum(axis2)


class TestFusionMultistageReductions:

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4, 5), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_multistage_reductions(self, xp):
        return lambda x: x.prod(axis=1).sum(axis=1)

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_multistage_reductions_and_elementwise(self, xp):
        return lambda x: (xp.sqrt(x).prod(axis=0) + x).sum(axis=1) * 2


class TestFusionMultistageReductionsMultiAxis:

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4, 5, 6), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_multistage_reductions(self, xp):
        return lambda x: x.prod(axis=(-1, 1)).sum(axis=(0, 1))


class TestFusionReductionRoutines:

    def generate_inputs(self, xp):
        x = testing.shaped_random((30,), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion()
    def test_sum(self, xp):
        return lambda x: xp.sum(x)

    @fusion_utils.check_fusion()
    def test_prod(self, xp):
        return lambda x: xp.prod(x)

    @fusion_utils.check_fusion()
    def test_nansum(self, xp):
        return lambda x: xp.nansum(x)

    @fusion_utils.check_fusion()
    def test_nanprod(self, xp):
        return lambda x: xp.nanprod(x)


class TestFusionMisc:

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @pytest.mark.skipif(
        not fusion_utils.can_use_grid_synchronization(),
        reason='Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_sum_div_clip(self, xp):
        def impl(x):
            x = x / xp.sum(x)
            return xp.clip(x, 2, 7)
        return impl

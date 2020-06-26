import unittest

import numpy

import cupy
from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(1,), (3, 4), (2, 1, 4), (2, 0, 3)],
    'axis': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
}))
class TestFusionReductionAxis(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random(self.shape, xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion(accept_error=numpy.AxisError)
    def test_sum_axis(self, xp):
        return lambda x: cupy.sum(x, self.axis)

    @fusion_utils.check_fusion(accept_error=numpy.AxisError)
    def test_sum_kwargs_axis(self, xp):
        return lambda x: cupy.sum(x, axis=self.axis)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(1,), (3, 4), (2, 1, 4), (2, 0, 3), (2, 3, 2, 2, 3)],
    'axis': [
        None, (0,), (1,), (0, 1), (1, 2), (0, 2), (1, 3)
        # TODO(asi1024): Fix core.simple_reduction_kernel to raise Error.
        # (0, 0), (-1, 1)
    ],
}))
class TestFusionReductionMultiAxis(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random(self.shape, xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion(accept_error=(ValueError, numpy.AxisError))
    def test_sum_axis(self, xp):
        return lambda x: cupy.sum(x, self.axis)

    @fusion_utils.check_fusion(accept_error=(ValueError, numpy.AxisError))
    def test_sum_kwargs_axis(self, xp):
        return lambda x: cupy.sum(x, axis=self.axis)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [
        (120, 128, 144),
        (119, 127, 143),
        (128, 128, 128),
        (32, 1024, 1024)
    ],
    'axis': [None, 0, 1, 2, (0, 1), (0, 2), (1, 2)],
}))
@testing.slow
class TestFusionReductionLarge(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random(self.shape, xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion()
    def test_sum_kwargs_axis(self, xp):
        return lambda x: cupy.sum(x, axis=self.axis)


# TODO(asi1024): Support for bool and complex dtypes.
@testing.gpu
class TestFusionReductionSpecifyDtype(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return (x,), {}

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True, no_complex=True)
    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_sum(self, xp, dtype1, dtype2):
        return lambda x: x.sum(axis=0, dtype=dtype2)


@testing.gpu
@testing.parameterize(*testing.product({
    'axis': [None, 0, 1],
}))
class TestFusionReductionAndElementwise(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return (x, y), {}

    @fusion_utils.check_fusion()
    def test_premap_one_array(self, xp):
        return lambda x, y: xp.sum(x * 3, self.axis)

    @fusion_utils.check_fusion()
    def test_premap_two_arrays(self, xp):
        return lambda x, y: xp.sum(x + y, self.axis)

    @fusion_utils.check_fusion()
    def test_postmap_one_array(self, xp):
        return lambda x, y: xp.sum(x, self.axis) + 3

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_postmap_two_arrays(self, xp):
        return lambda x, y: xp.sum(x, self.axis) + y

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_premap_postmap(self, xp):
        return lambda x, y: xp.sum(xp.sqrt(x) + y, self.axis) * 2 + y

    # TODO(asi1024): Uncomment after replace fusion implementaiton.
    # @fusion_utils.check_fusion()
    # def test_premap_inplace(self, xp):
    #     def impl(x, y):
    #         x += 2
    #         y += x
    #         return xp.sum(y, self.axis)
    #     return impl

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_postmap_inplace(self, xp):
        def impl(x, y):
            y += x
            res = xp.sum(x, self.axis)
            y += res
        return impl


@testing.gpu
@testing.parameterize(*testing.product({
    'axis1': [None, 0, 1],
    'axis2': [None, 0, 1],
}))
class TestFusionMultipleReductions(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return (x, y), {}

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_two_distinct_reductions(self, xp):
        return lambda x, y: (x.sum(self.axis1), y.sum(self.axis2))

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_two_reductions_and_elementwise(self, xp):
        return lambda x, y: x.sum(self.axis1) + y.sum(self.axis2)


@testing.gpu
class TestFusionMultistageReductions(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4, 5), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_multistage_reductions(self, xp):
        return lambda x: x.prod(axis=1).sum(axis=1)

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_multistage_reductions_and_elementwise(self, xp):
        return lambda x: (xp.sqrt(x).prod(axis=0) + x).sum(axis=1) * 2


@testing.gpu
class TestFusionMultistageReductionsMultiAxis(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4, 5, 6), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @unittest.skipUnless(
        fusion_utils.can_use_grid_synchronization(),
        'Requires CUDA grid synchronization')
    @fusion_utils.check_fusion()
    def test_multistage_reductions(self, xp):
        return lambda x: x.prod(axis=(-1, 1)).sum(axis=(0, 1))

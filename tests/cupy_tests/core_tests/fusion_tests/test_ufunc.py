import itertools
import unittest

from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


def _permutate_shapes(shapes_list):
    # Permutates input shapes
    permutated_shapes_set = set()
    for shapes in shapes_list:
        for permutated_shapes in itertools.permutations(shapes):
            permutated_shapes_set.add(permutated_shapes)
    return list(permutated_shapes_set)


@testing.gpu
@testing.parameterize(*testing.product({
    'shapes': _permutate_shapes([
        # Same shapes
        ((1,), (1,)),
        ((3, 4), (3, 4)),

        # Broadcast
        ((10,), (1,)),
        ((3, 4), (3, 1)),
        ((3, 4), (1, 4)),
        ((3, 4), (4,)),
        ((3, 4), (1, 1)),
        ((3, 4), (1,)),
        ((2, 3, 4), (1, 1, 1)),
        ((3, 1), (1, 4)),
        ((2, 1, 4), (3, 1)),

        # TODO(asi1024): Fix testing.shaped_random to support 0-dim array.
        # # 0-dim shape
        # ((), ()),
        # ((1,), ()),
        # ((3,), ()),
        # ((2, 3), ()),

        # 0-size shape
        ((0,), (0,)),
        ((0,), (1,)),
        ((2, 0, 3), (2, 0, 3)),
        ((2, 0, 3), (0, 1)),

        # Large case
        ((256, 256), (256,)),
        ((256, 256), (256, 1)),
    ])
}))
class TestFusionBroadcast(unittest.TestCase):

    def generate_inputs(self, xp):
        shape1, shape2 = self.shapes
        x = testing.shaped_random(shape1, xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random(shape2, xp, 'int64', scale=10, seed=1)
        return (x, y), {}

    @fusion_utils.check_fusion()
    def test_broadcast(self, xp):
        return lambda x, y: x + y

    # TODO(asi1024): Uncomment after replace fusion implementaiton.

    # @fusion_utils.check_fusion(accept_error=ValueError)
    # def test_broadcast_inplace(self, xp):
    #     def impl(x, y):
    #         x += y
    #     return impl


@testing.gpu
@testing.parameterize(*testing.product({
    'shapes': _permutate_shapes([
        ((2,), (3,)),
        ((2,), (0,)),
        ((3, 2), (3, 3)),
        ((3, 2), (2, 2)),
        ((3,), (1, 2)),
    ])
}))
class TestFusionBroadcastInvalid(unittest.TestCase):

    def generate_inputs(self, xp):
        shape1, shape2 = self.shapes
        x = testing.shaped_random(shape1, xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random(shape2, xp, 'int64', scale=10, seed=1)
        return (x, y), {}

    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_broadcast(self, xp):
        return lambda x, y: x + y

    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_broadcast_inplace(self, xp):
        def impl(x, y):
            x += y
        return impl


@testing.gpu
class TestFusionParseInput(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return (x,), {}

    @fusion_utils.check_fusion()
    def test_add(self, xp):
        return lambda x: x + x

    # TODO(asi1024): Should fix cupy.ufunc
    @fusion_utils.check_fusion(accept_error=(ValueError, TypeError))
    def test_add_too_less_param(self, xp):
        return lambda x: xp.add(x)

    # TODO(asi1024): Should fix cupy.ufunc
    @fusion_utils.check_fusion(accept_error=(ValueError, TypeError))
    def test_add_too_much_param(self, xp):
        return lambda x: xp.add(x, x, x, x)

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_none(self, xp):
        return lambda x: x + None

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_object(self, xp):
        return lambda x: x + object()

    # TODO(asi1024): Should fix cupy.ufunc
    # @fusion_utils.check_fusion()
    # def test_add_out_none(self, xp):
    #     def impl(x):
    #         xp.add(x, x, None)
    #         return x
    #     return impl

    @fusion_utils.check_fusion()
    def test_add_kwargs_out_none(self, xp):
        def impl(x):
            xp.add(x, x, out=None)
        return impl

    # TODO(asi1024): Should fix cupy.ufunc
    # @fusion_utils.check_fusion(accept_error=ValueError)
    # def test_add_both_out_none(self, xp):
    #     def impl(x):
    #         xp.add(x, x, None, out=None)
    #     return impl

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_out_object(self, xp):
        def impl(x):
            xp.add(x, x, object())
            return x
        return impl

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_kwargs_out_object(self, xp):
        def impl(x):
            xp.add(x, x, out=object())
            return x
        return impl

    @fusion_utils.check_fusion()
    def test_divmod(self, xp):
        return lambda x: xp.divmod(x, x)


@testing.gpu
class TestFusionOutDtype(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=1)
        z = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=2)
        return (x, y, z), {}

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), full=True, no_complex=True)
    @fusion_utils.check_fusion(accept_error=TypeError)
    @testing.with_requires('numpy>=1.13')
    def test_outarg(self, xp, dtype1, dtype2):
        def impl(x, y, z):
            xp.add(x, y, out=z)
            return z
        return impl


@testing.gpu
class TestFusionScalar(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return (array,), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_python_scalar_r(self, xp, dtype1, dtype2):
        def func(array):
            py_scalar = dtype2(1).item()
            return array + py_scalar

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_numpy_scalar_r(self, xp, dtype1, dtype2):
        def func(array):
            np_scalar = dtype2(1)
            return array + np_scalar

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_python_scalar_l(self, xp, dtype1, dtype2):
        def func(array):
            py_scalar = dtype2(1).item()
            return py_scalar + array

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_numpy_scalar_l(self, xp, dtype1, dtype2):
        def func(array):
            np_scalar = dtype2(1)
            return np_scalar + array

        return func

    def python_scalar_param_r(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        py_scalar = dtype2(1).item()
        return (array, py_scalar), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='python_scalar_param_r')
    def test_python_scalar_param_r(self, xp, dtype1, dtype2):
        def func(array, py_scalar):
            return array + py_scalar

        return func

    def python_scalar_param_l(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        py_scalar = dtype2(1).item()
        return (py_scalar, array), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='python_scalar_param_l')
    def test_python_scalar_param_l(self, xp, dtype1, dtype2):
        def func(py_scalar, array):
            return py_scalar + array

        return func

    def numpy_scalar_param_r(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        np_scalar = dtype2(1)
        return (array, np_scalar), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param_r')
    def test_numpy_scalar_param_r(self, xp, dtype1, dtype2):
        def func(array, np_scalar):
            return array + np_scalar

        return func

    def numpy_scalar_param_l(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        np_scalar = dtype2(1)
        return (np_scalar, array), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param_l')
    def test_numpy_scalar_param_l(self, xp, dtype1, dtype2):
        def func(np_scalar, array):
            return np_scalar + array

        return func

    def numpy_scalar_params_binop(self, xp, dtype1, dtype2):
        scalar1 = dtype1(1)
        scalar2 = dtype2(1)
        array = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return (scalar1, scalar2, array), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(
        generate_inputs_name='numpy_scalar_params_binop')
    def test_numpy_scalar_params_binop(self, xp, dtype1, dtype2):
        def func(scalar1, scalar2, array):
            dtype = (scalar1 + scalar2).dtype
            return array.astype(dtype)

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(
        generate_inputs_name='numpy_scalar_params_binop')
    def test_scalar_inplace_update(self, xp, dtype1, dtype2):
        def func(scalar1, scalar2, array):
            scalar1_copy = scalar1
            scalar1 += scalar2
            return array + scalar1 + scalar1_copy

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param_r')
    def test_scalar_inplace_update_with_array(self, xp, dtype1, dtype2):
        def func(array, scalar):
            scalar += array
            return scalar

        return func

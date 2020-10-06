import unittest

import numpy

from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


class FusionArrayTestBase(unittest.TestCase):

    def _get_argument(self, xp, dtype, seed, value_type):
        dtype = numpy.dtype(dtype)
        if value_type == 'array':
            x = testing.shaped_random((3, 4), xp, dtype, scale=5, seed=seed)
            # Avoid zero-division
            # TODO(imanishi): Doing this only in division tests.
            x[x == 0] = 1
            return x
        if value_type == 'scalar':
            return dtype.type(3)
        if value_type == 'primitive':
            return dtype.type(3).tolist()
        assert False

    def generate_inputs(self, xp, dtype1, dtype2):
        x = self._get_argument(xp, dtype1, 0, self.left_value)
        y = self._get_argument(xp, dtype2, 1, self.right_value)
        return (x, y), {}


@testing.gpu
@testing.parameterize(*testing._parameterized.product_dict(
    [
        {'name': 'neg', 'func': lambda x, y: -x},
        {'name': 'add', 'func': lambda x, y: x + y},
        {'name': 'sub', 'func': lambda x, y: x - y},
        {'name': 'mul', 'func': lambda x, y: x * y},
        {'name': 'div', 'func': lambda x, y: x / y},
        {'name': 'pow', 'func': lambda x, y: x ** y},
        {'name': 'eq', 'func': lambda x, y: x == y},
        {'name': 'ne', 'func': lambda x, y: x != y},
        {'name': 'lt', 'func': lambda x, y: x < y},
        {'name': 'le', 'func': lambda x, y: x <= y},
        {'name': 'gt', 'func': lambda x, y: x > y},
        {'name': 'ge', 'func': lambda x, y: x >= y},
    ],
    [
        {'left_value': 'array', 'right_value': 'array'},
        {'left_value': 'array', 'right_value': 'scalar'},
        {'left_value': 'array', 'right_value': 'primitive'},
        {'left_value': 'scalar', 'right_value': 'array'},
        {'left_value': 'primitive', 'right_value': 'array'},
    ]
))
class TestFusionArrayOperator(FusionArrayTestBase):

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True)
    @fusion_utils.check_fusion()
    def test_operator(self, xp, dtype1, dtype2):
        return self.func


@testing.gpu
@testing.parameterize(*testing._parameterized.product_dict(
    [
        {'name': 'lshift', 'func': lambda x, y: x << y},
        {'name': 'rshift', 'func': lambda x, y: x >> y},
        {'name': 'and', 'func': lambda x, y: x & y},
        {'name': 'or', 'func': lambda x, y: x | y},
        {'name': 'xor', 'func': lambda x, y: x ^ y},
        {'name': 'invert', 'func': lambda x, y: ~x},
    ],
    [
        {'left_value': 'array', 'right_value': 'array'},
        {'left_value': 'array', 'right_value': 'scalar'},
        {'left_value': 'array', 'right_value': 'primitive'},
        {'left_value': 'scalar', 'right_value': 'array'},
        {'left_value': 'primitive', 'right_value': 'array'},
    ]
))
class TestFusionArrayBitwiseOperator(FusionArrayTestBase):

    def _is_uint64(self, x):
        return not isinstance(x, int) and x.dtype == 'uint64'

    def _is_signed_int(self, x):
        return isinstance(x, int) or x.dtype.kind == 'i'

    @testing.for_int_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True)
    @fusion_utils.check_fusion()
    def test_operator(self, xp, dtype1, dtype2):
        def func(x, y):
            if ((self._is_uint64(x) and self._is_signed_int(y))
                    or (self._is_uint64(y) and self._is_signed_int(x))):
                # Skip TypeError case.
                return
            return self.func(x, y)

        return func


@testing.gpu
@testing.parameterize(
    {'left_value': 'array', 'right_value': 'array'},
    {'left_value': 'array', 'right_value': 'scalar'},
    {'left_value': 'array', 'right_value': 'primitive'},
    {'left_value': 'scalar', 'right_value': 'array'},
    {'left_value': 'primitive', 'right_value': 'array'},
)
class TestFusionArrayFloorDivide(FusionArrayTestBase):

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True, no_complex=True)
    @fusion_utils.check_fusion()
    def test_floor_divide(self, xp, dtype1, dtype2):
        return lambda x, y: x // y


# TODO(imanishi): Fix TypeError in use of dtypes_combination test.
@testing.gpu
@testing.parameterize(*testing._parameterized.product_dict(
    [
        {'left_value': 'array', 'right_value': 'array'},
        {'left_value': 'array', 'right_value': 'scalar'},
        {'left_value': 'array', 'right_value': 'primitive'},
    ]
))
class TestFusionArrayInplaceOperator(FusionArrayTestBase):

    def generate_inputs(self, xp, dtype):
        x = self._get_argument(xp, dtype, 0, self.left_value)
        y = self._get_argument(xp, dtype, 1, self.right_value)
        return (x, y), {}

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_iadd(self, xp, dtype):
        def func(x, y):
            x += y

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_isub(self, xp, dtype):
        def func(x, y):
            x -= y

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_imul(self, xp, dtype):
        def func(x, y):
            x *= y

        return func

    @testing.for_float_dtypes()
    @fusion_utils.check_fusion()
    def test_itruediv_py3(self, xp, dtype):
        def func(x, y):
            x /= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion(accept_error=(TypeError,))
    def test_int_itruediv_py3_raises(self, xp, dtype):
        def func(x, y):
            x /= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_imod(self, xp, dtype):
        def func(x, y):
            x %= y

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ipow(self, xp, dtype):
        def func(x, y):
            x **= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ilshift(self, xp, dtype):
        def func(x, y):
            x <<= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_irshift(self, xp, dtype):
        def func(x, y):
            x >>= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_iand(self, xp, dtype):
        def func(x, y):
            x &= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ior(self, xp, dtype):
        def func(x, y):
            x |= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ixor(self, xp, dtype):
        def func(x, y):
            x ^= y

        return func


@testing.gpu
class TestFusionArraySetItem(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int32', scale=10, seed=1)
        return (x, y), {}

    # TODO(imanishi): Fix TypeError in use of dtypes_combination test.
    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_setitem_ellipsis(self, xp):
        def func(x, y):
            y[...] = x
            return y

        return func

    # TODO(imanishi): Fix TypeError in use of dtypes_combination test.
    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_setitem_non_slice(self, xp):
        def func(x, y):
            y[:] = x
            return y

        return func


@testing.gpu
class TestFusionArrayMethods(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype, scale=10, seed=0)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_copy(self, xp, dtype):
        return lambda x: x.copy()

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_sum(self, xp, dtype):
        return lambda x: x.sum()

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_prod(self, xp, dtype):
        return lambda x: x.prod()

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_max(self, xp, dtype):
        return lambda x: x.max()

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_min(self, xp, dtype):
        return lambda x: x.min()

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_all(self, xp, dtype):
        return lambda x: x.all()

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_any(self, xp, dtype):
        return lambda x: x.any()


@testing.gpu
class TestFusionArrayAsType(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return (x,), {}

    # TODO(asi1024): Raise complex warnings.
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @fusion_utils.check_fusion()
    def test_astype(self, xp, dtype1, dtype2):
        return lambda x: x.astype(dtype2)

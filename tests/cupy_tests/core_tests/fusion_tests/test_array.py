from __future__ import annotations

import pytest

import numpy

from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


class FusionArrayTestBase:

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

    def generate_inputs(self, xp, dtype1, dtype2, left_value, right_value):
        x = self._get_argument(xp, dtype1, 0, left_value)
        y = self._get_argument(xp, dtype2, 1, right_value)
        return (x, y), {}


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x, y: -x, id='neg'),
        pytest.param(lambda x, y: x + y, id='add'),
        pytest.param(lambda x, y: x - y, id='sub'),
        pytest.param(lambda x, y: x * y, id='mul'),
        pytest.param(lambda x, y: x / y, id='div'),
        pytest.param(lambda x, y: x ** y, id='pow'),
        pytest.param(lambda x, y: x == y, id='eq'),
        pytest.param(lambda x, y: x != y, id='ne'),
        pytest.param(lambda x, y: x < y, id='lt'),
        pytest.param(lambda x, y: x <= y, id='le'),
        pytest.param(lambda x, y: x > y, id='gt'),
        pytest.param(lambda x, y: x >= y, id='ge'),
    ]
)
@pytest.mark.parametrize(
    "left_value,right_value",
    [
        ('array', 'array'),
        ('array', 'scalar'),
        ('array', 'primitive'),
        ('scalar', 'array'),
        ('primitive', 'array'),
    ]
)
class TestFusionArrayOperator(FusionArrayTestBase):

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True)
    @fusion_utils.check_fusion()
    def test_operator(self, xp, dtype1, dtype2, func, left_value, right_value):
        return func


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda x, y: x << y, id='lshift'),
        pytest.param(lambda x, y: x >> y, id='rshift'),
        pytest.param(lambda x, y: x & y, id='and'),
        pytest.param(lambda x, y: x | y, id='or'),
        pytest.param(lambda x, y: x ^ y, id='xor'),
        pytest.param(lambda x, y: ~x, id='invert'),
    ]
)
@pytest.mark.parametrize(
    "left_value,right_value",
    [
        ('array', 'array'),
        ('array', 'scalar'),
        ('array', 'primitive'),
        ('scalar', 'array'),
        ('primitive', 'array'),
    ]
)
class TestFusionArrayBitwiseOperator(FusionArrayTestBase):

    def _is_uint64(self, x):
        return not isinstance(x, int) and x.dtype == 'uint64'

    def _is_signed_int(self, x):
        return isinstance(x, int) or x.dtype.kind == 'i'

    @testing.for_int_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True)
    @fusion_utils.check_fusion()
    def test_operator(self, xp, dtype1, dtype2, func, left_value, right_value):
        def impl(x, y):
            if ((self._is_uint64(x) and self._is_signed_int(y))
                    or (self._is_uint64(y) and self._is_signed_int(x))):
                # Skip TypeError case.
                return
            return func(x, y)

        return impl


@pytest.mark.parametrize(
    "left_value,right_value",
    [
        ('array', 'array'),
        ('array', 'scalar'),
        ('array', 'primitive'),
        ('scalar', 'array'),
        ('primitive', 'array'),
    ]
)
class TestFusionArrayFloorDivide(FusionArrayTestBase):

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True, no_complex=True)
    @fusion_utils.check_fusion()
    def test_floor_divide(self, xp, dtype1, dtype2, left_value, right_value):
        return lambda x, y: x // y


# TODO(imanishi): Fix TypeError in use of dtypes_combination test.
@pytest.mark.parametrize(
    "left_value,right_value",
    [
        ('array', 'array'),
        ('array', 'scalar'),
        ('array', 'primitive'),
    ]
)
class TestFusionArrayInplaceOperator(FusionArrayTestBase):

    def generate_inputs(self, xp, dtype, left_value, right_value):
        x = self._get_argument(xp, dtype, 0, left_value)
        y = self._get_argument(xp, dtype, 1, right_value)
        return (x, y), {}

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_iadd(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x += y

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_isub(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x -= y

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_imul(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x *= y

        return func

    @testing.for_float_dtypes()
    @fusion_utils.check_fusion()
    def test_itruediv_py3(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x /= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion(accept_error=(TypeError,))
    def test_int_itruediv_py3_raises(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x /= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_imod(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x %= y

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ipow(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x **= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ilshift(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x <<= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_irshift(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x >>= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_iand(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x &= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ior(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x |= y

        return func

    @testing.for_int_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_ixor(self, xp, dtype, left_value, right_value):
        def func(x, y):
            x ^= y

        return func


class TestFusionArraySetItem:

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


class TestFusionArrayMethods:

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


class TestFusionArrayAsType:

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return (x,), {}

    # TODO(asi1024): Raise complex warnings.
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @fusion_utils.check_fusion()
    def test_astype(self, xp, dtype1, dtype2):
        return lambda x: x.astype(dtype2)

import unittest

import numpy
import pytest

import cupy
from cupy import testing


def _generate_type_routines_input(xp, dtype, obj_type):
    dtype = numpy.dtype(dtype)
    if obj_type == 'dtype':
        return dtype
    if obj_type == 'specifier':
        return str(dtype)
    if obj_type == 'scalar':
        return dtype.type(3)
    if obj_type == 'array':
        return xp.zeros(3, dtype=dtype)
    if obj_type == 'primitive':
        return type(dtype.type(3).tolist())
    assert False


@testing.parameterize(
    *testing.product({
        'obj_type': ['dtype', 'specifier', 'scalar', 'array', 'primitive'],
    })
)
class TestCanCast(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=('from_dtype', 'to_dtype'))
    @testing.numpy_cupy_equal()
    def test_can_cast(self, xp, from_dtype, to_dtype):
        from_obj = _generate_type_routines_input(xp, from_dtype, self.obj_type)
        ret = xp.can_cast(from_obj, to_dtype)
        assert isinstance(ret, bool)
        return ret


class TestCommonType(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_common_type_empty(self, xp):
        ret = xp.common_type()
        assert type(ret) == type
        return ret

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_equal()
    def test_common_type_single_argument(self, xp, dtype):
        array = _generate_type_routines_input(xp, dtype, 'array')
        ret = xp.common_type(array)
        assert type(ret) == type
        return ret

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True)
    @testing.numpy_cupy_equal()
    def test_common_type_two_arguments(self, xp, dtype1, dtype2):
        array1 = _generate_type_routines_input(xp, dtype1, 'array')
        array2 = _generate_type_routines_input(xp, dtype2, 'array')
        ret = xp.common_type(array1, array2)
        assert type(ret) == type
        return ret

    @testing.for_all_dtypes()
    def test_common_type_bool(self, dtype):
        for xp in (numpy, cupy):
            array1 = _generate_type_routines_input(xp, dtype, 'array')
            array2 = _generate_type_routines_input(xp, 'bool_', 'array')
            with pytest.raises(TypeError):
                xp.common_type(array1, array2)


@testing.parameterize(
    *testing.product({
        'obj_type1': ['dtype', 'specifier', 'scalar', 'array', 'primitive'],
        'obj_type2': ['dtype', 'specifier', 'scalar', 'array', 'primitive'],
    })
)
class TestResultType(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_equal()
    def test_result_type(self, xp, dtype1, dtype2):
        input1 = _generate_type_routines_input(xp, dtype1, self.obj_type1)
        input2 = _generate_type_routines_input(xp, dtype2, self.obj_type2)
        ret = xp.result_type(input1, input2)
        assert isinstance(ret, numpy.dtype)
        return ret

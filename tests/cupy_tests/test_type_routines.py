import unittest

import numpy

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
        return xp.can_cast(from_obj, to_dtype)


class TestCommonType(unittest.TestCase):

    # NumPy 1.9 cannot handle float16 in ``numpy.common_type``.
    @testing.with_requires('numpy>=1.10')
    @testing.for_dtypes_combination('efdFD', names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_equal()
    def test_common_type(self, xp, dtype1, dtype2):
        array1 = _generate_type_routines_input(xp, dtype1, 'array')
        array2 = _generate_type_routines_input(xp, dtype2, 'array')
        return xp.common_type(array1, array2)


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
        return xp.result_type(input1, input2)

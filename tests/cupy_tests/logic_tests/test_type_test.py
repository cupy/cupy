import unittest

import numpy

from cupy import testing


class TestIsScalar(testing.NumpyAliasBasicTestBase):

    func = 'isscalar'


@testing.parameterize(
    *testing.product({
        'value': [
            0, 0.0, True,
            numpy.int32(1), numpy.array([1, 2], numpy.int32),
            numpy.complex(1), numpy.complex(1j), numpy.complex(1 + 1j),
            None, object(), 'abc', '', int, numpy.int32]}))
class TestIsScalarValues(testing.NumpyAliasValuesTestBase):

    func = 'isscalar'

    def setUp(self):
        self.args = (self.value,)


@testing.parameterize(
    *testing.product({
        'value': [
            # C and F
            numpy.ones(24, order='C'),
            # C and not F
            numpy.ones((4, 6), order='C'),
            # not C and F
            numpy.ones((4, 6), order='F'),
            # not C and not F
            numpy.ones((4, 6), order='C')[1:3][1:3],
        ]
    })
)
class TestIsFortran(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test(self, xp):
        return xp.isfortran(self.value)


class TypeTestingTestBase(object):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test(self, xp, dtype):
        return getattr(xp, self.func)(xp.ones(5, dtype=dtype))


class TypeTestingObjTestBase(object):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test(self, xp, dtype):
        return getattr(xp, self.func)(xp.ones(5, dtype=dtype))


class TestIsComplex(TypeTestingTestBase, unittest.TestCase):

    func = 'iscomplex'


class TestIsComplexObj(TypeTestingObjTestBase, unittest.TestCase):

    func = 'iscomplexobj'


class TestIsReal(TypeTestingTestBase, unittest.TestCase):

    func = 'isreal'


class TestIsRealObj(TypeTestingObjTestBase, unittest.TestCase):

    func = 'isrealobj'

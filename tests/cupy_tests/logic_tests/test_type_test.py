import unittest

import numpy

from cupy import testing


class TestIsScalar(testing.NumpyAliasBasicTestBase):

    func = 'isscalar'

    @testing.with_requires('numpy>=1.18')
    def test_argspec(self):
        super().test_argspec()


@testing.parameterize(
    *testing.product({
        'value': [
            0, 0.0, True,
            numpy.int32(1), numpy.array([1, 2], numpy.int32),
            numpy.complex128(1), numpy.complex128(1j),
            numpy.complex128(1 + 1j),
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
        return xp.isfortran(xp.asarray(self.value))


@testing.parameterize(
    {'func': 'iscomplex'},
    {'func': 'isreal'},
)
class TestTypeTestingFunctions(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test(self, xp, dtype):
        return getattr(xp, self.func)(xp.ones(5, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_scalar(self, xp, dtype):
        return getattr(xp, self.func)(dtype(3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_list(self, xp, dtype):
        return getattr(xp, self.func)(
            testing.shaped_arange((2, 3), xp, dtype).tolist())


@testing.parameterize(
    {'func': 'iscomplexobj'},
    {'func': 'isrealobj'},
)
class TestTypeTestingObjFunctions(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test(self, xp, dtype):
        return getattr(xp, self.func)(xp.ones(5, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_scalar(self, xp, dtype):
        return getattr(xp, self.func)(dtype(3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_list(self, xp, dtype):
        return getattr(xp, self.func)(
            testing.shaped_arange((2, 3), xp, dtype).tolist())

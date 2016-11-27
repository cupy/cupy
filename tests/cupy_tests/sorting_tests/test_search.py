import unittest

import numpy

from cupy import testing


@testing.gpu
class TestSearch(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.argmax(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_argmax_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.argmax(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmin()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_argmin_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return a.argmin()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.argmin(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.argmin(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=2)


@testing.parameterize(
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (4,),      'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (3, 4)},
    {'cond_shape': (3, 4),    'x_shape': (2, 3, 4), 'y_shape': (4,)},
)
@testing.gpu
class TestWhereTwoArrays(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['cond_type', 'x_type', 'y_type'])
    @testing.numpy_cupy_allclose()
    def test_where_two_arrays(self, xp, cond_type, x_type, y_type):
        m = testing.shaped_random(self.cond_shape, xp, xp.bool_)
        # Almost all values of a matrix `shaped_random` makes are not zero.
        # To make a sparse matrix, we need multiply `m`.
        cond = testing.shaped_random(self.cond_shape, xp, cond_type) * m
        x = testing.shaped_random(self.x_shape, xp, x_type)
        y = testing.shaped_random(self.y_shape, xp, y_type)
        return xp.where(cond, x, y)


@testing.parameterize(
    {'cond_shape': (2, 3, 4)},
    {'cond_shape': (4,)},
    {'cond_shape': (2, 3, 4)},
    {'cond_shape': (3, 4)},
)
@testing.gpu
class TestWhereCond(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_where_cond(self, xp, dtype):
        m = testing.shaped_random(self.cond_shape, xp, xp.bool_)
        cond = testing.shaped_random(self.cond_shape, xp, dtype) * m
        return xp.where(cond)


@testing.gpu
class TestWhereError(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_one_argument(self, xp):
        cond = testing.shaped_random((3, 4), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, xp.int32)
        xp.where(cond, x)


@testing.parameterize(
    {"array": numpy.random.randint(0, 2, (20,))},
    {"array": numpy.random.randn(3, 2, 4)},
    {"array": numpy.array(0)},
    {"array": numpy.array(1)},
)
@testing.gpu
class TestNonzero(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_nonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.nonzero(array)


@testing.parameterize(
    {"array": numpy.random.randint(0, 2, (20,))},
    {"array": numpy.random.randn(3, 2, 4)},
    {"array": numpy.array(0)},
    {"array": numpy.array(1)},
)
@testing.gpu
class TestFlatNonzero(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flatnonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.flatnonzero(array)

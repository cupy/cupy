import unittest

import numpy
import six

from cupy import testing


def calc_out_shape(shape, axis, keepdims):
    if axis is None:
        axis = list(six.moves.range(len(shape)))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    shape = numpy.array(shape)

    if keepdims:
        shape[axis] = 1
    else:
        shape[axis] = -1
        shape = filter(lambda x: x != -1, shape)
    return tuple(shape)


@testing.gpu
class TestAll(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.f = 'all'

    def check(self, x, xp, axis):
        a = getattr(xp, self.f)(x, axis)

        out_shape = calc_out_shape(x.shape, axis, False)
        b = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, axis, out=b)

        c = getattr(xp, self.f)(x, axis, keepdims=True)

        out_shape2 = calc_out_shape(x.shape, axis, True)
        d = xp.empty(out_shape2, dtype=x.dtype)
        getattr(xp, self.f)(x, axis, out=d, keepdims=True)
        return (a, b, c, d)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_reduce(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype) - 10
        return self.check(x, xp, None)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_reduce_2(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype) - 10
        return self.check(x, xp, (0, 1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_reduce_all_zero(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype=dtype)
        return self.check(x, xp, None)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_reduce_all_zero_2(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype=dtype)
        return self.check(x, xp, (0, 1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_reduce_all_one(self, xp, dtype):
        x = xp.ones((2, 3, 4), dtype=dtype)
        return self.check(x, xp, None)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_reduce_all_one_2(self, xp, dtype):
        x = xp.ones((2, 3, 4), dtype=dtype)
        return self.check(x, xp, (0, 1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype) - 10
        return self.check(x, xp, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_2(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype) - 10
        return self.check(x, xp, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_3(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype) - 10
        return self.check(x, xp, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_4(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype) - 10
        return self.check(x, xp, (0, 1))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_all_zero(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype=dtype)
        return self.check(x, xp, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_2_all_zero(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype=dtype)
        return self.check(x, xp, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_3_all_zero(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype=dtype)
        return self.check(x, xp, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_4_all_zero(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype=dtype)
        return self.check(x, xp, (0, 1))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_all_one(self, xp, dtype):
        x = xp.ones((2, 3, 4), dtype=dtype)
        return self.check(x, xp, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_2_all_one(self, xp, dtype):
        x = xp.ones((2, 3, 4), dtype=dtype)
        return self.check(x, xp, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_3_all_one(self, xp, dtype):
        x = xp.ones((2, 3, 4), dtype=dtype)
        return self.check(x, xp, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_all_partial_reduce_4_all_one(self, xp, dtype):
        x = xp.ones((2, 3, 4), dtype=dtype)
        return self.check(x, xp, (0, 1))


@testing.gpu
class TestAny(TestAll):

    def setUp(self):
        super(TestAny, self).setUp()
        self.f = 'any'

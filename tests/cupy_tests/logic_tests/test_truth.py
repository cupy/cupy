import numpy

from cupy import testing


def _calc_out_shape(shape, axis, keepdims):
    if axis is None:
        axis = list(range(len(shape)))
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


@testing.parameterize(
    *testing.product(
        {'f': ['all', 'any'],
         'x': [numpy.arange(24).reshape(2, 3, 4) - 10,
               numpy.zeros((2, 3, 4)),
               numpy.ones((2, 3, 4)),
               numpy.zeros((0, 3, 4)),
               numpy.ones((0, 3, 4))],
         'axis': [None, (0, 1, 2), 0, 1, 2, (0, 1)],
         'keepdims': [False, True]}))
@testing.gpu
class TestAllAny:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(xp, self.f)(x, self.axis, None, self.keepdims)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, self.axis, out, self.keepdims)
        return out


@testing.parameterize(
    *testing.product(
        {'f': ['all', 'any'],
         'x': [numpy.array([[[numpy.nan]]]),
               numpy.array([[[numpy.nan, 0]]]),
               numpy.array([[[numpy.nan, 1]]]),
               numpy.array([[[numpy.nan, 0, 1]]])],
         'axis': [None, (0, 1, 2), 0, 1, 2, (0, 1)],
         'keepdims': [False, True]}))
@testing.gpu
class TestAllAnyWithNaN:

    @testing.for_dtypes(
        (numpy.float64, numpy.float32, numpy.float16, numpy.bool_))
    @testing.numpy_cupy_array_equal()
    def test_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(xp, self.f)(x, self.axis, None, self.keepdims)

    @testing.for_dtypes(
        (numpy.float64, numpy.float32, numpy.float16, numpy.bool_))
    @testing.numpy_cupy_array_equal()
    def test_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, self.axis, out, self.keepdims)
        return out


class TestAllAnyAlias:
    @testing.numpy_cupy_array_equal()
    def test_alltrue(self, xp):
        return xp.alltrue(xp.array([1, 2, 3]))

    @testing.numpy_cupy_array_equal()
    def test_sometrue(self, xp):
        return xp.sometrue(xp.array([0]))


@testing.parameterize(
    *testing.product(
        {'f': ['in1d', 'isin'],
         'shape_x': [
             (0, ),
             (3, ),
             (2, 3),
             (2, 1, 3),
             (2, 0, 1),
             (2, 0, 1, 1)
        ],
            'shape_y': [
             (0, ),
             (3, ),
             (2, 3),
             (2, 1, 3),
             (2, 0, 1),
             (2, 0, 1, 1)
        ],
            'assume_unique': [False, True],
            'invert': [False, True]}))
@testing.gpu
class TestIn1DIsIn:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test(self, xp, dtype):
        x = testing.shaped_arange(self.shape_x, xp, dtype)
        y = testing.shaped_arange(self.shape_y, xp, dtype)
        if xp is numpy and self.f == 'isin':
            return xp.in1d(x, y, self.assume_unique, self.invert)\
                .reshape(x.shape)
        return getattr(xp, self.f)(x, y, self.assume_unique, self.invert)


class TestSetdiff1d:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_same_arrays(self, xp, dtype):
        x = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        y = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        return xp.setdiff1d(x, y, assume_unique=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_diff_size_arr_inputs(self, xp, dtype):
        x = xp.array([3, 4, 9, 1, 5, 4], dtype=dtype)
        y = xp.array([8, 7, 3, 9, 0], dtype=dtype)
        return xp.setdiff1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_diff_elements(self, xp, dtype):
        x = xp.array([3, 4, 9, 1, 5, 4], dtype=dtype)
        y = xp.array([8, 7, 3, 9, 0], dtype=dtype)
        return xp.setdiff1d(x, y, assume_unique=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_with_2d(self, xp, dtype):
        x = testing.shaped_random((2, 3), xp, dtype=dtype)
        y = testing.shaped_random((3, 5), xp, dtype=dtype)
        return xp.setdiff1d(x, y, assume_unique=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_with_duplicate_elements(self, xp, dtype):
        x = xp.array([1, 2, 3, 2, 2, 6], dtype=dtype)
        y = xp.array([3, 4, 2, 1, 1, 9], dtype=dtype)
        return xp.setdiff1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_empty_arr(self, xp, dtype):
        x = xp.array([], dtype=dtype)
        y = xp.array([], dtype=dtype)
        return xp.setdiff1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_more_dim(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4, 8), xp, dtype=dtype)
        y = testing.shaped_arange((5, 4, 2), xp, dtype=dtype)
        return xp.setdiff1d(x, y, assume_unique=True)

    @testing.numpy_cupy_array_equal()
    def test_setdiff1d_bool_val(self, xp):
        x = xp.array([True, False, True])
        y = xp.array([False])
        return xp.setdiff1d(x, y)


class TestSetxor1d:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_same_arrays(self, xp, dtype):
        x = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        y = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        return xp.setxor1d(x, y, assume_unique=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_diff_size_arr_inputs(self, xp, dtype):
        x = xp.array([3, 4, 9, 1, 5, 4], dtype=dtype)
        y = xp.array([8, 7, 3, 9, 0], dtype=dtype)
        return xp.setxor1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_diff_elements(self, xp, dtype):
        x = xp.array([3, 4, 9, 1, 5, 4], dtype=dtype)
        y = xp.array([8, 7, 3, 9, 0], dtype=dtype)
        return xp.setxor1d(x, y, assume_unique=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_with_2d(self, xp, dtype):
        x = testing.shaped_random((2, 3), xp, dtype=dtype)
        y = testing.shaped_random((3, 5), xp, dtype=dtype)
        return xp.setxor1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_with_duplicate_elements(self, xp, dtype):
        x = xp.array([1, 2, 3, 2, 2, 6], dtype=dtype)
        y = xp.array([3, 4, 2, 1, 1, 9], dtype=dtype)
        return xp.setxor1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_empty_arr(self, xp, dtype):
        x = xp.array([], dtype=dtype)
        y = xp.array([], dtype=dtype)
        return xp.setxor1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setxor1d_more_dim(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4, 8), xp, dtype=dtype)
        y = testing.shaped_arange((5, 4, 2), xp, dtype=dtype)
        return xp.setxor1d(x, y)

    @testing.numpy_cupy_array_equal()
    def test_setxor1d_bool_val(self, xp):
        x = xp.array([True, False, True])
        y = xp.array([False])
        return xp.setxor1d(x, y)


class TestIntersect1d:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_one_dim_with_unique_values(self, xp, dtype):
        a = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        b = xp.array([1, 2, 3, 4, 5], dtype=dtype)
        return xp.intersect1d(a, b, assume_unique=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_with_random_val(self, xp, dtype):
        a = xp.array([3, 4, 9, 1, 5, 4], dtype=dtype)
        b = xp.array([8, 7, 3, 9, 0], dtype=dtype)
        return xp.intersect1d(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_more_dim(self, xp, dtype):
        a = testing.shaped_random((3, 4), xp, dtype=dtype)
        b = testing.shaped_random((5, 2), xp, dtype=dtype)
        return xp.intersect1d(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_return_indices(self, xp, dtype):
        a = xp.array([2, 3, 4, 1, 9, 4], dtype=dtype)
        b = xp.array([7, 5, 1, 2, 9, 3], dtype=dtype)
        return xp.intersect1d(a, b, return_indices=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_multiple_instances(self, xp, dtype):
        a = xp.array([2, 4, 5, 2, 1, 5], dtype=dtype)
        b = xp.array([4, 6, 2, 5, 7, 6], dtype=dtype)
        return xp.intersect1d(a, b, return_indices=True)


class TestUnion1d:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_union1d(self, xp, dtype):
        x = xp.array([4, 1, 1, 1, 9, 9, 9], dtype=dtype)
        y = xp.array([4, 0, 5, 2, 0, 0, 5], dtype=dtype)
        return xp.union1d(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_union1d_2(self, xp, dtype):
        x = testing.shaped_arange((5, 2), xp, dtype=dtype)
        y = testing.shaped_arange((2, 3, 4), xp, dtype=dtype)
        return xp.union1d(x, y)

    @testing.numpy_cupy_array_equal()
    def test_union1d_3(self, xp):
        x = xp.zeros((2, 2), dtype=xp.complex_)
        y = xp.array([[1+1j, 2+3j], [4+1j, 0+7j]])
        return xp.union1d(x, y)

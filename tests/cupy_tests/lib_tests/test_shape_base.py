import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy.exceptions import AxisError


@testing.parameterize(*(testing.product({'axis': [0, 1, -1]})))
class TestApplyAlongAxis(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_simple(self, xp):
        a = xp.ones((20, 10), 'd')
        return xp.apply_along_axis(len, self.axis, a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_3d(self, xp, dtype):
        a = xp.arange(27, dtype=dtype).reshape((3, 3, 3))
        return xp.apply_along_axis(xp.sum, self.axis, a)

    @testing.numpy_cupy_array_equal()
    def test_0d_array(self, xp):

        def sum_to_0d(x):
            """ Sum x, returning a 0d array of the same class """
            assert x.ndim == 1
            return xp.squeeze(xp.sum(x, keepdims=True))

        a = xp.ones((6, 3))
        return xp.apply_along_axis(sum_to_0d, self.axis, a)

    @testing.numpy_cupy_array_equal()
    def test_axis_insertion_2d(self, xp):

        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert x.ndim == 1
            return (x[::-1] * x[1:, None])

        # 2d insertion
        a2d = xp.arange(6 * 3).reshape((6, 3))
        return xp.apply_along_axis(f1to2, self.axis, a2d)

    @testing.numpy_cupy_array_equal()
    def test_axis_insertion_3d(self, xp):

        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert x.ndim == 1
            return (x[::-1] * x[1:, None])

        # 3d insertion
        a3d = xp.arange(6 * 5 * 3).reshape((6, 5, 3))
        return xp.apply_along_axis(f1to2, self.axis, a3d)

    def test_empty1(self):
        # can't apply_along_axis when there's no chance to call the function
        def never_call(x):
            assert False  # should never be reached

        for xp in [numpy, cupy]:
            a = xp.empty((0, 0))
            with pytest.raises(ValueError):
                xp.apply_along_axis(never_call, self.axis, a)

    def test_empty2(self):
        # but it's sometimes ok with some non-zero dimensions
        def empty_to_1(x):
            assert len(x) == 0
            return 1

        for xp in [numpy, cupy]:
            shape = [10, 10]
            shape[self.axis] = 0
            shape = tuple(shape)
            a = xp.empty(shape)
            if self.axis == 0:
                other_axis = 1
            else:
                other_axis = 0
            with pytest.raises(ValueError):
                xp.apply_along_axis(empty_to_1, other_axis, a)

            # okay to call along the shape 0 axis
            testing.assert_array_equal(
                xp.apply_along_axis(empty_to_1, self.axis, a),
                xp.ones((10,))
            )

    @testing.numpy_cupy_array_equal()
    def test_tuple_outs(self, xp):
        def func(x):
            return x.sum(axis=-1), x.prod(axis=-1), x.max(axis=-1)

        a = testing.shaped_arange((2, 2, 2), xp, cupy.int64)
        return xp.apply_along_axis(func, 1, a)


@testing.with_requires('numpy>=1.16')
def test_apply_along_axis_invalid_axis():
    for xp in [numpy, cupy]:
        a = xp.ones((8, 4))
        for axis in [-3, 2]:
            with pytest.raises(AxisError):
                xp.apply_along_axis(xp.sum, axis, a)


class TestPutAlongAxis(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_put_along_axis_empty(self, xp, dtype):
        a = xp.array([], dtype=dtype).reshape(0, 10)
        i = xp.array([], dtype=xp.int64).reshape(0, 10)
        vals = xp.array([]).reshape(0, 10)
        ret = xp.put_along_axis(a, i, vals, axis=0)
        assert ret is None
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_simple(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        indices_max = xp.argmax(a, axis=0, keepdims=True)
        ret = xp.put_along_axis(a, indices_max, 0, axis=0)
        assert ret is None
        return a

    @testing.for_all_dtypes()
    def test_indices_values_arr_diff_dims(self, dtype):
        for xp in [numpy, cupy]:
            a = testing.shaped_arange((3, 3, 3), xp, dtype)
            i_max = xp.argmax(a, axis=0, keepdims=False)
            with pytest.raises(ValueError):
                xp.put_along_axis(a, i_max, -99, axis=1)


@testing.parameterize(*testing.product({
    'axis': [0, 1],
}))
class TestPutAlongAxes(unittest.TestCase):

    def test_replace_max(self):
        arr = cupy.array([[10, 30, 20], [60, 40, 50]])
        indices_max = cupy.argmax(arr, axis=self.axis, keepdims=True)
        # replace the max with a small value
        cupy.put_along_axis(arr, indices_max, -99, axis=self.axis)
        # find the new minimum, which should max
        indices_min = cupy.argmin(arr, axis=self.axis, keepdims=True)
        testing.assert_array_equal(indices_min, indices_max)


class TestPutAlongAxisNone(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_axis_none(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        i = xp.array([1, 3])
        val = xp.array([99, 100])
        ret = xp.put_along_axis(a, i, val, axis=None)
        assert ret is None
        return a

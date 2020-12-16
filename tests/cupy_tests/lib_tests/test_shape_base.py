import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(*(testing.product({'axis': [0, 1, -1]})))
@testing.gpu
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


@testing.gpu
@testing.with_requires('numpy>=1.16')
def test_apply_along_axis_invalid_axis():
    for xp in [numpy, cupy]:
        a = xp.ones((8, 4))
        for axis in [-3, 2]:
            with pytest.raises(numpy.AxisError):
                xp.apply_along_axis(xp.sum, axis, a)

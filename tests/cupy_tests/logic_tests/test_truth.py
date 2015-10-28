import unittest

import numpy
import six

from cupy import testing


def _calc_out_shape(shape, axis, keepdims):
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


@testing.parameterize(
    *testing.product(
        {'f': ['all', 'any'],
         'x': [numpy.arange(24).reshape(2, 3, 4) - 10,
               numpy.zeros((2, 3, 4)),
               numpy.ones((2, 3, 4))],
         'axis': [None, (0, 1, 2), 0, 1, 2, (0, 1)],
         'keepdims': [False, True]}))
@testing.gpu
class TestAllAny(unittest.TestCase):

    _multiprocess_can_split_ = True

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
class TestAllAnyWithNaN(unittest.TestCase):

    _multiprocess_can_split_ = True

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

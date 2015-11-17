import unittest

import numpy

from cupy import testing


# Note that numpy.bincount does not support uint64 as it casts an input array to
# int64.
_except_uint64 = (
    numpy.float16, numpy.float32, numpy.float64,
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32,
    numpy.bool_)


def for_dtypes_except_uint64(name='dtype'):
    return testing.for_dtypes(_except_uint64, name=name)


def for_dtypes_combination_except_uint64(names):
    return testing.helper.for_dtypes_combination(_except_uint64, names=names)


@testing.gpu
class TestHistogram(unittest.TestCase):

    _multiprocess_can_split_ = True

    @for_dtypes_except_uint64()
    @testing.numpy_cupy_allclose()
    def test_bincount(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x)

    @for_dtypes_except_uint64()
    @testing.numpy_cupy_allclose()
    def test_bincount_duplicated_value(self, xp, dtype):
        x = xp.array([1, 2, 2, 1, 2, 4], dtype)
        return xp.bincount(x)

    @for_dtypes_combination_except_uint64(names=['x_type', 'w_type'])
    @testing.numpy_cupy_allclose()
    def test_bincount_with_weight(self, xp, x_type, w_type):
        x = testing.shaped_arange((3,), xp, x_type)
        w = testing.shaped_arange((3,), xp, w_type)
        return xp.bincount(x, weights=w)

    @for_dtypes_except_uint64()
    @testing.numpy_cupy_allclose()
    def test_bincount_with_minlength(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=5)

    @for_dtypes_combination_except_uint64(names=['x_type', 'w_type'])
    @testing.numpy_cupy_raises()
    def test_bincount_invalid_weight_length(self, xp, x_type, w_type):
        x = testing.shaped_arange((1,), xp, x_type)
        w = testing.shaped_arange((2,), xp, w_type)
        return xp.bincount(x, weights=w)

    @testing.for_signed_dtypes()
    @testing.numpy_cupy_raises()
    def test_bincount_negative(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype) - 2
        return xp.bincount(x)

    @for_dtypes_except_uint64()
    @testing.numpy_cupy_raises()
    def test_bincount_too_deep(self, xp, dtype):
        x = xp.array([[1]], dtype)
        return xp.bincount(x)

    @for_dtypes_except_uint64()
    @testing.numpy_cupy_raises()
    def test_bincount_too_small(self, xp, dtype):
        x = xp.zeros((), dtype)
        return xp.bincount(x)

    @for_dtypes_except_uint64()
    @testing.numpy_cupy_raises()
    def test_bincount_zero_minlength(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=0)

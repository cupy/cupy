import sys
import unittest

import numpy

from cupy import testing


# Note that numpy.bincount does not support uint64 on 64-bit environment
# as it casts an input array to intp.
# And it does not support uint32, int64 and uint64 on 32-bit environment.
_all_types = (
    numpy.float16, numpy.float32, numpy.float64,
    numpy.int8, numpy.int16, numpy.int32,
    numpy.uint8, numpy.uint16,
    numpy.bool_)
_signed_types = (
    numpy.int8, numpy.int16, numpy.int32,
    numpy.bool_)

if sys.maxsize > 2 ** 32:
    _all_types = _all_types + (numpy.int64, numpy.uint32)
    _signed_types = _signed_types + (numpy.int64,)


def for_all_dtypes_bincount(name='dtype'):
    return testing.for_dtypes(_all_types, name=name)


def for_signed_dtypes_bincount(name='dtype'):
    return testing.for_dtypes(_signed_types, name=name)


def for_all_dtypes_combination_bincount(names):
    return testing.helper.for_dtypes_combination(_all_types, names=names)


@testing.gpu
class TestHistogram(unittest.TestCase):

    @testing.with_requires('numpy>=1.15.0')
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_list_equal()
    def test_histogram(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        y, bin_edges = xp.histogram(x)
        return y, bin_edges

    @testing.with_requires('numpy>=1.15.0')
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_list_equal()
    def test_histogram_same_value(self, xp, dtype):
        x = xp.zeros(10, dtype)
        y, bin_edges = xp.histogram(x, 3)
        return y, bin_edges

    @testing.with_requires('numpy>=1.15.0')
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_list_equal()
    def test_histogram_empty(self, xp, dtype):
        x = xp.array([], dtype)
        y, bin_edges = xp.histogram(x)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_list_equal()
    def test_histogram_int_bins(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        y, bin_edges = xp.histogram(x, 4)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_list_equal()
    def test_histogram_array_bins(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        bins = testing.shaped_arange((3,), xp, dtype)
        y, bin_edges = xp.histogram(x, bins)
        return y, bin_edges

    @testing.with_requires('numpy>=1.13.2')
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_histogram_bins_not_ordered(self, xp, dtype):
        # numpy 1.13.1 does not check this error correctly with unsigned int.
        x = testing.shaped_arange((10,), xp, dtype)
        bins = xp.array([1, 3, 2], dtype)
        xp.histogram(x, bins)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_duplicated_value(self, xp, dtype):
        x = xp.array([1, 2, 2, 1, 2, 4], dtype)
        return xp.bincount(x)

    @for_all_dtypes_combination_bincount(names=['x_type', 'w_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_with_weight(self, xp, x_type, w_type):
        x = testing.shaped_arange((3,), xp, x_type)
        w = testing.shaped_arange((3,), xp, w_type)
        return xp.bincount(x, weights=w)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_with_minlength(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=5)

    @for_all_dtypes_combination_bincount(names=['x_type', 'w_type'])
    @testing.numpy_cupy_raises()
    def test_bincount_invalid_weight_length(self, xp, x_type, w_type):
        x = testing.shaped_arange((1,), xp, x_type)
        w = testing.shaped_arange((2,), xp, w_type)
        return xp.bincount(x, weights=w)

    @for_signed_dtypes_bincount()
    @testing.numpy_cupy_raises()
    def test_bincount_negative(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype) - 2
        return xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_raises()
    def test_bincount_too_deep(self, xp, dtype):
        x = xp.array([[1]], dtype)
        return xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_raises()
    def test_bincount_too_small(self, xp, dtype):
        x = xp.zeros((), dtype)
        return xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_zero(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=0)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_raises()
    def test_bincount_too_small_minlength(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=-1)

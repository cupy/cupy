import sys
import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator


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
    return testing.for_dtypes_combination(_all_types, names=names)


@testing.gpu
class TestHistogram(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        y, bin_edges = xp.histogram(x)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_same_value(self, xp, dtype):
        x = xp.zeros(10, dtype)
        y, bin_edges = xp.histogram(x, 3)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_density(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        y, bin_edges = xp.histogram(x, density=True)
        # check normalization
        area = xp.sum(y * xp.diff(bin_edges))
        testing.assert_allclose(area, 1)
        return y, bin_edges

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_histogram_range_lower_outliers(self, xp, dtype):
        # Check that lower outliers are not tallied
        a = xp.arange(10, dtype=dtype) + .5
        h, b = xp.histogram(a, range=[0, 9])
        assert int(h.sum()) == 9
        return h, b

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_histogram_range_upper_outliers(self, xp, dtype):
        # Check that upper outliers are not tallied
        a = xp.arange(10, dtype=dtype) + .5
        h, b = xp.histogram(a, range=[1, 10])
        assert int(h.sum()) == 9
        return h, b

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_histogram_range_with_density(self, xp, dtype):
        a = xp.arange(10, dtype=dtype) + .5
        h, b = xp.histogram(a, range=[1, 9], density=True)
        # check normalization
        testing.assert_allclose(float((h * xp.diff(b)).sum()), 1)
        return h

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_histogram_range_with_weights_and_density(self, xp, dtype):
        a = xp.arange(10, dtype=dtype) + .5
        w = xp.arange(10, dtype=dtype) + .5
        h, b = xp.histogram(a, range=[1, 9], weights=w, density=True)
        testing.assert_allclose(float((h * xp.diff(b)).sum()), 1)
        return h

    def test_histogram_invalid_range(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                # range must be None or have two elements
                xp.histogram(xp.arange(10), range=[1, 9, 15])

    def test_histogram_invalid_range2(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.histogram(xp.arange(10), range=10)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_histogram_weights_mismatch(self, dtype):
        for xp in (numpy, cupy):
            a = xp.arange(10, dtype=dtype) + .5
            w = xp.arange(11, dtype=dtype) + .5
            with pytest.raises(ValueError):
                xp.histogram(a, range=[1, 9], weights=w, density=True)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_histogram_int_weights_dtype(self, xp, dtype):
        # Check the type of the returned histogram
        a = xp.arange(10, dtype=dtype)
        h, b = xp.histogram(a, weights=xp.ones(10, int))
        assert xp.issubdtype(h.dtype, xp.integer)
        return h

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_histogram_float_weights_dtype(self, xp, dtype):
        # Check the type of the returned histogram
        a = xp.arange(10, dtype=dtype)
        h, b = xp.histogram(a, weights=xp.ones(10, float))
        assert xp.issubdtype(h.dtype, xp.floating)
        return h

    def test_histogram_weights_basic(self):
        v = cupy.random.rand(100)
        w = cupy.ones(100) * 5
        a, b = cupy.histogram(v)
        na, nb = cupy.histogram(v, density=True)
        wa, wb = cupy.histogram(v, weights=w)
        nwa, nwb = cupy.histogram(v, weights=w, density=True)
        testing.assert_array_almost_equal(a * 5, wa)
        testing.assert_array_almost_equal(na, nwa)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_histogram_float_weights(self, xp, dtype):
        # Check weights are properly applied.
        v = xp.linspace(0, 10, 10, dtype=dtype)
        w = xp.concatenate((xp.zeros(5, dtype=dtype), xp.ones(5, dtype=dtype)))
        wa, wb = xp.histogram(v, bins=xp.arange(11), weights=w)
        testing.assert_array_almost_equal(wa, w)
        return wb

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_histogram_int_weights(self, xp, dtype):
        # Check with integer weights
        v = xp.asarray([1, 2, 2, 4], dtype=dtype)
        w = xp.asarray([4, 3, 2, 1], dtype=dtype)
        wa, wb = xp.histogram(v, bins=4, weights=w)
        testing.assert_array_equal(wa, [4, 5, 0, 1])
        return wa, wb

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_histogram_int_weights_normalized(self, xp, dtype):
        v = xp.asarray([1, 2, 2, 4], dtype=dtype)
        w = xp.asarray([4, 3, 2, 1], dtype=dtype)
        wa, wb = xp.histogram(v, bins=4, weights=w, density=True)
        testing.assert_array_almost_equal(
            wa, xp.asarray([4, 5, 0, 1]) / 10. / 3. * 4)
        return wb

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_int_weights_nonuniform_bins(self, xp, dtype):
        # Check weights with non-uniform bin widths
        a, b = xp.histogram(
            xp.arange(9, dtype=dtype),
            xp.asarray([0, 1, 3, 6, 10], dtype=dtype),
            weights=xp.asarray([2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dtype),
            density=True)
        testing.assert_array_almost_equal(a, [.2, .1, .1, .075])
        return a, b

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_histogram_complex_weights(self, xp, dtype):
        values = xp.asarray([1.3, 2.5, 2.3])
        weights = xp.asarray([1, -1, 2]) + 1j * xp.asarray([2, 1, 2])
        weights = weights.astype(dtype)
        a, b = xp.histogram(
            values, bins=2, weights=weights)
        return a, b

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_histogram_complex_weights_uneven_bins(self, xp, dtype):
        values = xp.asarray([1.3, 2.5, 2.3])
        weights = xp.asarray([1, -1, 2]) + 1j * xp.asarray([2, 1, 2])
        weights = weights.astype(dtype)
        a, b = xp.histogram(
            values, bins=xp.asarray([0, 2, 3]), weights=weights)
        return a, b

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_empty(self, xp, dtype):
        x = xp.array([], dtype)
        y, bin_edges = xp.histogram(x)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_int_bins(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        y, bin_edges = xp.histogram(x, 4)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_array_bins(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        bins = testing.shaped_arange((3,), xp, dtype)
        y, bin_edges = xp.histogram(x, bins)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_numpy_bins(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        bins = testing.shaped_arange((3,), numpy, dtype)
        y, bin_edges = xp.histogram(x, bins)
        return y, bin_edges

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_list_bins(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)
        bins = list(testing.shaped_arange((3,), numpy, dtype))
        y, bin_edges = xp.histogram(x, bins)
        return y, bin_edges

    # numpy 1.13.1 does not check this error correctly with unsigned int.
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_histogram_bins_not_ordered(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((10,), xp, dtype)
            bins = xp.array([1, 3, 2], dtype)
            with pytest.raises(ValueError):
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
    def test_bincount_invalid_weight_length(self, x_type, w_type):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((1,), xp, x_type)
            w = testing.shaped_arange((2,), xp, w_type)
            # TODO(imanishi): Split this test into a test for ValueError and
            # a test for TypeError.
            with pytest.raises((ValueError, TypeError)):
                xp.bincount(x, weights=w)

    @for_signed_dtypes_bincount()
    def test_bincount_negative(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3,), xp, dtype) - 2
            with pytest.raises(ValueError):
                xp.bincount(x)

    @for_all_dtypes_bincount()
    def test_bincount_too_deep(self, dtype):
        for xp in (numpy, cupy):
            x = xp.array([[1]], dtype)
            with pytest.raises(ValueError):
                xp.bincount(x)

    @for_all_dtypes_bincount()
    def test_bincount_too_small(self, dtype):
        for xp in (numpy, cupy):
            x = xp.zeros((), dtype)
            with pytest.raises(ValueError):
                xp.bincount(x)

    @for_all_dtypes_bincount()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_bincount_zero(self, xp, dtype):
        x = testing.shaped_arange((3,), xp, dtype)
        return xp.bincount(x, minlength=0)

    @for_all_dtypes_bincount()
    def test_bincount_too_small_minlength(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3,), xp, dtype)
            # TODO(imanishi): Split this test into a test for ValueError and
            # a test for TypeError.
            with pytest.raises((ValueError, TypeError)):
                xp.bincount(x, minlength=-1)


# This class compares CUB results against NumPy's
@testing.gpu
@unittest.skipUnless(cupy.cuda.cub.available, 'The CUB routine is not enabled')
class TestCubHistogram(unittest.TestCase):

    def setUp(self):
        self.old_accelerators = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators(['cub'])

    def tearDown(self):
        _accelerator.set_routine_accelerators(self.old_accelerators)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram(self, xp, dtype):
        x = testing.shaped_arange((10,), xp, dtype)

        if xp is numpy:
            return xp.histogram(x)

        # xp is cupy, first ensure we really use CUB
        cub_func = 'cupy._statistics.histogram.cub.device_histogram'
        with testing.AssertFunctionIsCalled(cub_func):
            xp.histogram(x)
        # ...then perform the actual computation
        return xp.histogram(x)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_range_float(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        h, b = xp.histogram(a, testing.shaped_arange((10,), xp, numpy.float64))
        assert int(h.sum()) == 10
        return h, b

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'],
                                        no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_with_bins(self, xp, dtype_a, dtype_b):
        x = testing.shaped_arange((10,), xp, dtype_a)
        bins = testing.shaped_arange((4,), xp, dtype_b)

        if xp is numpy:
            return xp.histogram(x, bins)[0]

        # xp is cupy, first ensure we really use CUB
        cub_func = 'cupy._statistics.histogram.cub.device_histogram'
        with testing.AssertFunctionIsCalled(cub_func):
            xp.histogram(x, bins)
        # ...then perform the actual computation
        return xp.histogram(x, bins)[0]

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'],
                                        no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_histogram_with_bins2(self, xp, dtype_a, dtype_b):
        x = testing.shaped_arange((10,), xp, dtype_a)
        bins = testing.shaped_arange((4,), xp, dtype_b)

        if xp is numpy:
            return xp.histogram(x, bins)[1]

        # xp is cupy, first ensure we really use CUB
        cub_func = 'cupy._statistics.histogram.cub.device_histogram'
        with testing.AssertFunctionIsCalled(cub_func):
            xp.histogram(x, bins)
        # ...then perform the actual computation
        return xp.histogram(x, bins)[1]


@testing.gpu
@testing.parameterize(*testing.product(
    {'bins': [
        # Test monotonically increasing with in-bounds values
        [1.5, 2.5, 4.0, 6.0],
        # Explicit out-of-bounds for x values
        [-1.0, 1.0, 2.5, 4.0, 20.0],
        # Repeated values should yield right-most or left-most indexes
        [0.0, 1.0, 1.0, 4.0, 4.0, 10.0],
    ],
        'increasing': [True, False],
        'right': [True, False],
        'shape': [(), (10,), (6, 3, 3)]})
)
class TestDigitize(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_digitize(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        bins = self.bins
        if not self.increasing:
            bins = bins[::-1]
        bins = xp.array(bins)
        y = xp.digitize(x, bins, right=self.right)
        return y,


@testing.gpu
@testing.parameterize(
    {'right': True},
    {'right': False})
class TestDigitizeNanInf(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = xp.array([1.0, 3.0, 5.0, 8.0, 12.0], xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float32)
        bins = xp.array([1.0, 3.0, 5.0, 8.0, float('nan')], xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins_repeated(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [1.0, 3.0, 5.0, 8.0, float('nan'), float('nan')]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins_decreasing(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [float('nan'), 8.0, 5.0, 3.0, 1.0]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_digitize_nan_bins_decreasing_repeated(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [float('nan'), float('nan'), float('nan'), 5.0, 3.0, 1.0]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_digitize_all_nan_bins(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float32)
        x[5] = float('nan')
        bins = [float('nan'), float('nan'), float('nan'), float('nan')]
        bins = xp.array(bins, xp.float32)
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_inf(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float64)
        x[5] = float('inf')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.digitize(x, bins, right=self.right)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_minf(self, xp):
        x = testing.shaped_arange((14,), xp, xp.float64)
        x[5] = float('-inf')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.digitize(x, bins, right=self.right)
        return y,


@testing.gpu
class TestDigitizeInvalid(unittest.TestCase):

    def test_digitize_complex(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((14,), xp, complex)
            bins = xp.array([1.0, 3.0, 5.0, 8.0, 12.0], complex)
            with pytest.raises(TypeError):
                xp.digitize(x, bins)

    def test_digitize_nd_bins(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((14,), xp, xp.float64)
            bins = xp.array([[1], [2]])
            with pytest.raises(ValueError):
                xp.digitize(x, bins)


@testing.parameterize(
    *testing.product(
        {'weights': [None, 1, 2],
         'weights_dtype': [numpy.int32, numpy.float64],
         'density': [True, False],
         'bins': [10, (8, 16, 12), (16, 8, 12), (16, 12, 8), (12, 8, 16),
                  'array_list'],
         'range': [None, ((20, 50), (10, 100), (0, 40))]}
    )
)
@testing.gpu
class TestHistogramdd(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-7, rtol=1e-7)
    def test_histogramdd(self, xp, dtype):
        x = testing.shaped_random((100, 3), xp, dtype, scale=100)
        if self.bins == 'array_list':
            bins = [xp.arange(0, 100, 4),
                    xp.arange(0, 100, 10),
                    xp.arange(25)]
        else:
            bins = self.bins
        if self.weights is not None:
            weights = xp.ones((x.shape[0],), dtype=self.weights_dtype)
        else:
            weights = None
        y, bin_edges = xp.histogramdd(x, bins=bins, range=self.range,
                                      weights=weights, density=self.density)
        return [y, ] + [e for e in bin_edges]


@testing.gpu
class TestHistogramddErrors(unittest.TestCase):

    def test_histogramdd_invalid_bins(self):
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            bins = [xp.arange(0, 100, 10), ] * 3
            with pytest.raises(ValueError):
                y, bin_edges = xp.histogramdd(x, bins)

    def test_histogramdd_invalid_bins2(self):
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            with pytest.raises(ValueError):
                y, bin_edges = xp.histogramdd(x, bins=0)

    def test_histogramdd_invalid_bins3(self):
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            bins = xp.arange(100)
            bins[30] = 99  # non-ascending bins
            with pytest.raises(ValueError):
                y, bin_edges = xp.histogramdd(x, bins=bins)

    def test_histogramdd_invalid_bins4(self):
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            bins = xp.arange(64).reshape((8, 8))  # too many dimensions
            with pytest.raises(ValueError):
                y, bin_edges = xp.histogramdd(x, bins=bins)

    def test_histogramdd_invalid_range(self):
        for xp in (numpy, cupy):
            x = testing.shaped_random((16, 2), xp, scale=100)
            r = ((0, 100),) * 3
            with pytest.raises(ValueError):
                y, bin_edges = xp.histogramdd(x, range=r)

    def test_histogramdd_disallow_arraylike_bins(self):
        x = testing.shaped_random((16, 2), cupy, scale=100)
        bins = [[0, 10, 20, 50, 90]] * 2  # too many dimensions
        with pytest.raises(ValueError):
            y, bin_edges = cupy.histogramdd(x, bins=bins)


@testing.parameterize(
    *testing.product(
        {'weights': [None, 1, 2],
         'weights_dtype': [numpy.int32, numpy.float64],
         'density': [True, False],
         'bins': [10, (8, 16), (16, 8), 'array_list', 'array'],
         'range': [None, ((20, 50), (10, 100))]}
    )
)
@testing.gpu
class TestHistogram2d(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-7, rtol=1e-7)
    def test_histogram2d(self, xp, dtype):
        x = testing.shaped_random((100, ), xp, dtype, scale=100)
        y = testing.shaped_random((100, ), xp, dtype, scale=100)
        if self.bins == 'array_list':
            bins = [xp.arange(0, 100, 4), xp.arange(0, 100, 10)]
        elif self.bins == 'array':
            bins = xp.arange(0, 100, 4)
        else:
            bins = self.bins
        if self.weights is not None:
            weights = xp.ones((x.shape[0],), dtype=self.weights_dtype)
        else:
            weights = None
        y, edges0, edges1 = xp.histogram2d(x, y, bins=bins,
                                           range=self.range, weights=weights,
                                           density=self.density)
        return y, edges0, edges1


@testing.gpu
class TestHistogram2dErrors(unittest.TestCase):

    def test_histogram2d_disallow_arraylike_bins(self):
        x = testing.shaped_random((16, ), cupy, scale=100)
        y = testing.shaped_random((16, ), cupy, scale=100)
        bins = [0, 10, 20, 50, 90]
        with pytest.raises(ValueError):
            y, bin_edges = cupy.histogram2d(x, y, bins=bins)

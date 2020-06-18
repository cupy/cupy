import unittest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


class FilterTestCaseBase(unittest.TestCase):
    """
    Add some utility methods for the parameterized tests for filters. these
    assume there are the "parameters" self.filter, self.wdtype or self.dtype,
    and self.ndim, self.kshape, or self.shape. Other optional "parameters" are
    also used if available like self.footprint when the filter is a filter
    that uses the footprint. These methods allow testing across multiple
    filter types much more easily.
    """

    # default param values if not provided
    filter = 'convolve'
    shape = (4, 5)
    ksize = 3
    dtype = numpy.float64
    footprint = True

    # Params that need no processing and just go into kwargs
    KWARGS_PARAMS = ('output', 'axis', 'mode', 'cval')

    def _filter(self, xp, scp):
        """
        The function that all tests end up calling, possibly after a few
        adjustments to the class "parameters".
        """
        # The filter function
        filter = getattr(scp.ndimage, self.filter)

        # The kwargs to pass to the filter function
        kwargs = {param: getattr(self, param)
                  for param in FilterTestCaseBase.KWARGS_PARAMS
                  if hasattr(self, param)}
        if hasattr(self, 'origin'):
            kwargs['origin'] = self._origin

        # The array we are filtering
        arr = testing.shaped_random(self.shape, xp, self.dtype)

        # The weights we are using to filter
        wghts = self._get_weights(xp)
        if isinstance(wghts, tuple) and len(wghts) == 2 and wghts[0] is None:
            # w is actually a tuple of (None, footprint)
            wghts, kwargs['footprint'] = wghts

        # Actually perform filtering
        return filter(arr, wghts, **kwargs)

    def _get_weights(self, xp):
        # Gets the second argument to the filter functions.
        # For convolve/correlate/convolve1d/correlate1d this is the weights.
        # For minimum_filter1d/maximum_filter1d this is the kernel size.
        #
        # For minimum_filter/maximum_filter this is a bit more complicated and
        # is either the kernel size or a tuple of None and the footprint. The
        # _filter() method knows about this and handles it automatically.

        if self.filter in ('convolve', 'correlate'):
            return testing.shaped_random(self._kshape, xp, self._dtype)

        if self.filter in ('convolve1d', 'correlate1d'):
            return testing.shaped_random((self.ksize,), xp, self._dtype)

        if self.filter in ('minimum_filter', 'maximum_filter'):
            if not self.footprint:
                return self.ksize
            kshape = self._kshape
            footprint = testing.shaped_random(kshape, xp, scale=1) > 0.5
            if not footprint.any():
                footprint = xp.ones(kshape)
            return None, footprint

        if self.filter in ('minimum_filter1d', 'maximum_filter1d'):
            return self.ksize

        raise RuntimeError('unsupported filter name')

    @property
    def _dtype(self):
        return getattr(self, 'wdtype', None) or self.dtype

    @property
    def _ndim(self):
        return getattr(self, 'ndim', len(getattr(self, 'shape', [])))

    @property
    def _kshape(self):
        return getattr(self, 'kshape', (self.ksize,) * self._ndim)

    @property
    def _origin(self):
        origin = getattr(self, 'origin', 0)
        if origin is not None:
            return origin
        is_1d = self.filter.endswith('1d')
        return -1 if is_1d else (-1, 1, -1, 1)[:self._ndim]


# Parameters common across all modes (with some overrides)
COMMON_PARAMS = {
    'shape': [(4, 5), (3, 4, 5), (1, 3, 4, 5)],
    'ksize': [3, 4],
    'dtype': [numpy.int32, numpy.float64],
}


# The bulk of the tests are done with this class
@testing.parameterize(*(
    testing.product([
        # Filter-function specific params
        testing.product({
            'filter': ['convolve', 'correlate'],
        }) + testing.product({
            'filter': ['convolve1d', 'correlate1d',
                       'minimum_filter1d', 'maximum_filter1d'],
            'axis': [0, 1, -1],
        }) + testing.product({
            'filter': ['minimum_filter', 'maximum_filter'],
            'footprint': [False, True],
        }),

        # Mode-specific params
        testing.product({
            **COMMON_PARAMS,
            'mode': ['reflect'],
            # With reflect test some of the other parameters as well
            'origin': [0, 1, None],
            'output': [None, numpy.int32, numpy.float64],
            'dtype': [numpy.uint8, numpy.int16, numpy.int32,
                      numpy.float32, numpy.float64],
        }) + testing.product({
            **COMMON_PARAMS,
            'mode': ['constant'], 'cval': [-1.0, 0.0, 1.0],
        }) + testing.product({
            **COMMON_PARAMS,
            'mode': ['nearest', 'wrap'],
        }) + testing.product({
            **COMMON_PARAMS,
            'shape': [(4, 5), (3, 4, 5)],  # no (1,3,4,5) due to scipy bug
            'mode': ['mirror'],
        })
    ])
))
@testing.gpu
@testing.with_requires('scipy')
class TestFilter(FilterTestCaseBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_filter(self, xp, scp):
        if self.dtype == getattr(self, 'output', None):
            raise unittest.SkipTest("redundant")
        return self._filter(xp, scp)


# Tests things requiring scipy >= 1.5.0
@testing.parameterize(*(
    testing.product([
        # Filter-function specific params
        testing.product({
            'filter': ['convolve', 'correlate'],
        }) + testing.product({
            'filter': ['convolve1d', 'correlate1d',
                       'minimum_filter1d', 'maximum_filter1d'],
            'axis': [0, 1, -1],
        }) + testing.product({
            'filter': ['minimum_filter', 'maximum_filter'],
            'footprint': [False, True],
        }),

        # Mode-specific params
        testing.product({
            **COMMON_PARAMS,
            'shape': [(1, 3, 4, 5)],
            'mode': ['mirror'],
        })
    ])
))
@testing.gpu
# SciPy behavior fixed in 1.5.0: https://github.com/scipy/scipy/issues/11661
@testing.with_requires('scipy>=1.5.0')
class TestMirrorWithDim1(FilterTestCaseBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_filter(self, xp, scp):
        return self._filter(xp, scp)


# Tests with weight dtypes that are distinct from the input and output dtypes
@testing.parameterize(*(
    testing.product([
        testing.product({
            'filter': ['convolve', 'correlate'],
        }) + testing.product({
            'filter': ['convolve1d', 'correlate1d'],
            'axis': [0, 1, -1],
        }),
        testing.product({
            **COMMON_PARAMS,
            'mode': ['reflect'],
            'output': [None, numpy.int32, numpy.float64],
            'dtype': [numpy.uint8, numpy.int16, numpy.int32,
                      numpy.float32, numpy.float64],
            'wdtype': [numpy.int32, numpy.float64],
        })
    ])
))
@testing.gpu
@testing.with_requires('scipy')
class TestWeightDtype(FilterTestCaseBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_filter(self, xp, scp):
        if self.dtype == self.wdtype:
            raise unittest.SkipTest("redundant")
        return self._filter(xp, scp)


# Tests special weights (ND)
@testing.parameterize(*testing.product({
    'filter': ['convolve', 'correlate', 'minimum_filter', 'maximum_filter'],
    'shape': [(3, 3), (3, 3, 3)],
    'dtype': [numpy.int32, numpy.float64],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSpecialWeightCases(FilterTestCaseBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_extra_0_dim(self, xp, scp):
        # NOTE: minimum/maximum_filter raise ValueError but convolve/correlate
        # return an array of zeroes the same shape as the input. This will
        # handle both and only pass is both numpy and cupy do the same thing.
        self.kshape = (0,) + self.shape
        return self._filter(xp, scp)

    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=RuntimeError)
    def test_missing_dim(self, xp, scp):
        self.kshape = self.shape[1:]
        return self._filter(xp, scp)

    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=RuntimeError)
    def test_extra_dim(self, xp, scp):
        self.kshape = self.shape[:1] + self.shape
        return self._filter(xp, scp)

    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=(RuntimeError,
                                                               ValueError))
    def test_replace_dim_with_0(self, xp, scp):
        self.kshape = (0,) + self.shape[1:]
        return self._filter(xp, scp)


# Tests special weights (1D)
@testing.parameterize(*testing.product({
    'filter': ['convolve1d', 'correlate1d',
               'minimum_filter1d', 'maximum_filter1d'],
    'shape': [(3, 3), (3, 3, 3)],
    'dtype': [numpy.int32, numpy.float64],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSpecialCases1D(FilterTestCaseBase):
    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=RuntimeError)
    def test_0_dim(self, xp, scp):
        self.ksize = 0
        return self._filter(xp, scp)


# Tests invalid axis value
@testing.parameterize(*testing.product({
    'filter': ['convolve1d', 'correlate1d',
               'minimum_filter1d', 'maximum_filter1d'],
    'shape': [(4, 5), (3, 4, 5), (1, 3, 4, 5)],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestInvalidAxis(FilterTestCaseBase):
    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=ValueError)
    def test_invalid_axis_pos(self, xp, scp):
        self.axis = len(self.shape)
        try:
            return self._filter(xp, scp)
        except numpy.AxisError:
            # numpy.AxisError is a subclass of ValueError
            # currently cupyx is raising numpy.AxisError but scipy is still
            # raising ValueError
            raise ValueError('invalid axis')

    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=ValueError)
    def test_invalid_axis_neg(self, xp, scp):
        self.axis = -len(self.shape) - 1
        try:
            return self._filter(xp, scp)
        except numpy.AxisError:
            raise ValueError('invalid axis')


# Tests invalid mode value
@testing.parameterize(*testing.product({
    'filter': ['convolve', 'correlate',
               'convolve1d', 'correlate1d',
               'minimum_filter', 'maximum_filter',
               'minimum_filter1d', 'maximum_filter1d'],
    'mode': ['unknown'],
    'shape': [(4, 5)],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestInvalidMode(FilterTestCaseBase):
    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=RuntimeError)
    def test_invalid_mode(self, xp, scp):
        return self._filter(xp, scp)


# Tests invalid origin values
@testing.parameterize(*testing.product({
    'filter': ['convolve', 'correlate',
               'convolve1d', 'correlate1d',
               'minimum_filter', 'maximum_filter',
               'minimum_filter1d', 'maximum_filter1d'],
    'ksize': [3, 4],
    'shape': [(4, 5)], 'dtype': [numpy.float64],
}))
@testing.gpu
# SciPy behavior fixed in 1.2.0: https://github.com/scipy/scipy/issues/822
@testing.with_requires('scipy>=1.2.0')
class TestInvalidOrigin(FilterTestCaseBase):
    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=ValueError)
    def test_invalid_origin_neg(self, xp, scp):
        self.origin = -self.ksize // 2 - 1
        return self._filter(xp, scp)

    @testing.numpy_cupy_raises(scipy_name='scp', accept_error=ValueError)
    def test_invalid_origin_pos(self, xp, scp):
        self.origin = self.ksize - self.ksize // 2
        return self._filter(xp, scp)

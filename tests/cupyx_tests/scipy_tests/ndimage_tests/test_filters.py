import unittest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass

# ######### Testing convolve and correlate ##########


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'mode': ['reflect'],
        'cval': [0.0],
        'origin': [0, 1, None],
        'adtype': [numpy.int8, numpy.int16, numpy.int32,
                   numpy.float32, numpy.float64],
        'wdtype': [None, numpy.int32, numpy.float64],
        'output': [None, numpy.int32, numpy.float64],
        'filter': ['convolve', 'correlate']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'mode': ['constant'],
        'cval': [-1.0, 0.0, 1.0],
        'origin': [0],
        'adtype': [numpy.int32, numpy.float64],
        'wdtype': [None],
        'output': [None],
        'filter': ['convolve', 'correlate']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'mode': ['nearest', 'mirror', 'wrap'],
        'cval': [0.0],
        'origin': [0],
        'adtype': [numpy.int32, numpy.float64],
        'wdtype': [None],
        'output': [None],
        'filter': ['convolve', 'correlate']
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveAndCorrelate(unittest.TestCase):

    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[:a.ndim]
        else:
            origin = self.origin
        return filter(a, w, output=self.output, mode=self.mode,
                      cval=self.cval, origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve_and_correlate(self, xp, scp):
        if 1 in self.shape and self.mode == 'mirror':
            raise unittest.SkipTest("requires scipy>1.5.0, tested later")
        if self.adtype == self.wdtype or self.adtype == self.output:
            raise unittest.SkipTest("redundant")
        a = testing.shaped_random(self.shape, xp, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, wdtype)
        return self._filter(xp, scp, a, w)


@testing.parameterize(*testing.product({
    'shape': [(1, 2, 3, 4)],
    'ksize': [3, 4],
    'dtype': [numpy.int32, numpy.float64],
    'filter': ['convolve', 'correlate']
}))
@testing.gpu
# SciPy behavior fixed in 1.5.0: https://github.com/scipy/scipy/issues/11661
@testing.with_requires('scipy>=1.5.0')
class TestConvolveAndCorrelateMirrorDim1(unittest.TestCase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve_and_correlate(self, xp, scp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, self.dtype)
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, output=None, mode='mirror', cval=0.0, origin=0)


@testing.parameterize(*testing.product({
    'ndim': [2, 3],
    'dtype': [numpy.int32, numpy.float64],
    'filter': ['convolve', 'correlate']
}))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveAndCorrelateSpecialCases(unittest.TestCase):

    def _filter(self, scp, a, w, mode='reflect', origin=0):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, mode=mode, origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_weights_with_size_zero_dim(self, xp, scp):
        a = testing.shaped_random((3, ) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((0, ) + (3, ) * self.ndim, xp, self.dtype)
        return self._filter(scp, a, w)

    def test_invalid_shape_weights(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3, ) * (self.ndim - 1), cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)
        w = testing.shaped_random((0, ) + (3, ) * (self.ndim - 1), cupy,
                                  self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)

    def test_invalid_mode(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w, mode='unknown')

    # SciPy behavior fixed in 1.2.0: https://github.com/scipy/scipy/issues/822
    @testing.with_requires('scipy>=1.2.0')
    def test_invalid_origin(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        for lenw in [3, 4]:
            w = testing.shaped_random((lenw, ) * self.ndim, cupy, self.dtype)
            for origin in range(-3, 4):
                if (lenw // 2 + origin < 0) or (lenw // 2 + origin >= lenw):
                    with self.assertRaises(ValueError):
                        self._filter(cupyx.scipy, a, w, origin=origin)
                else:
                    self._filter(cupyx.scipy, a, w, origin=origin)


# ######### Testing convolve1d and correlate1d ##########


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'axis': [0, 1, -1],
        'mode': ['reflect'],
        'cval': [0.0],
        'origin': [0, 1, -1],
        'adtype': [numpy.int8, numpy.int16, numpy.int32,
                   numpy.float32, numpy.float64],
        'wdtype': [None, numpy.int32, numpy.float64],
        'output': [None, numpy.int32, numpy.float64],
        'filter': ['convolve1d', 'correlate1d']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'axis': [0, 1, -1],
        'mode': ['constant'],
        'cval': [-1.0, 0.0, 1.0],
        'origin': [0],
        'adtype': [numpy.int32, numpy.float64],
        'wdtype': [None],
        'output': [None],
        'filter': ['convolve1d', 'correlate1d']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'axis': [0, 1, -1],
        'mode': ['nearest', 'mirror', 'wrap'],
        'cval': [0.0],
        'origin': [0],
        'adtype': [numpy.int32, numpy.float64],
        'wdtype': [None],
        'output': [None],
        'filter': ['convolve1d', 'correlate1d']
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolve1DAndCorrelate1D(unittest.TestCase):

    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, axis=self.axis, output=self.output, mode=self.mode,
                      cval=self.cval, origin=self.origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve1d_and_correlate1d(self, xp, scp):
        if 1 in self.shape and self.mode == 'mirror':
            raise unittest.SkipTest("requires scipy>1.5.0, tested later")
        if self.adtype == self.wdtype or self.adtype == self.output:
            raise unittest.SkipTest("redundant")
        a = testing.shaped_random(self.shape, xp, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,), xp, wdtype)
        return self._filter(xp, scp, a, w)


@testing.parameterize(*testing.product({
    'shape': [(1, 2, 3, 4)],
    'ksize': [3, 4],
    'axis': [0, 1, -1],
    'dtype': [numpy.int32, numpy.float64],
    'filter': ['convolve1d', 'correlate1d']
}))
@testing.gpu
# SciPy behavior fixed in 1.5.0: https://github.com/scipy/scipy/issues/11661
@testing.with_requires('scipy>=1.5.0')
class TestConvolveAndCorrelateMirrorDim1(unittest.TestCase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve_and_correlate(self, xp, scp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, self.dtype)
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, axis=self.axis, output=None, mode='mirror',
                      cval=0.0, origin=0)


@testing.parameterize(*testing.product({
    'ndim': [2, 3],
    'dtype': [numpy.int32, numpy.float64],
    'filter': ['convolve1d', 'correlate1d']
}))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolve1DAndCorrelate1DSpecialCases(unittest.TestCase):

    def _filter(self, scp, a, w, mode='reflect', origin=0):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, mode=mode, origin=origin)

    def test_weights_with_size_zero_dim(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((0, 3), cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)

    def test_invalid_shape_weights(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3, 3), cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)
        w = testing.shaped_random((0, ), cupy,
                                  self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w)

    def test_invalid_mode(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3,), cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyx.scipy, a, w, mode='unknown')

    # SciPy behavior fixed in 1.2.0: https://github.com/scipy/scipy/issues/822
    @testing.with_requires('scipy>=1.2.0')
    def test_invalid_origin(self):
        a = testing.shaped_random((3, ) * self.ndim, cupy, self.dtype)
        for lenw in [3, 4]:
            w = testing.shaped_random((lenw, ), cupy, self.dtype)
            for origin in range(-3, 4):
                if (lenw // 2 + origin < 0) or (lenw // 2 + origin >= lenw):
                    with self.assertRaises(ValueError):
                        self._filter(cupyx.scipy, a, w, origin=origin)
                else:
                    self._filter(cupyx.scipy, a, w, origin=origin)


# ######### Testing minimum_filter and maximum_filter ##########

@testing.parameterize(*testing.product({
    'size': [3, 4],
    'footprint': [None, 'random'],
    'mode': ['reflect', 'constant', 'nearest', 'mirror', 'wrap'],
    'origin': [0, None],
    'x_dtype': [numpy.int32, numpy.float32],
    'output': [None, numpy.float64],
    'filter': ['minimum_filter', 'maximum_filter']
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMinimumMaximumFilter(unittest.TestCase):

    shape = (4, 5)
    cval = 0.0

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[:x.ndim]
        else:
            origin = self.origin
        if self.footprint is None:
            size, footprint = self.size, None
        else:
            size = None
            shape = (self.size, ) * x.ndim
            footprint = testing.shaped_random(shape, xp, scale=1) > .5
            if not footprint.any():
                footprint = xp.ones(shape)
        return filter(x, size=size, footprint=footprint,
                      output=self.output, mode=self.mode, cval=self.cval,
                      origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_minimum_and_maximum_filter(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.x_dtype)
        return self._filter(xp, scp, x)


# ######### Testing minimum_filter1d and maximum_filter1d ##########


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'axis': [0, 1, -1],
        'mode': ['reflect'],
        'cval': [0.0],
        'origin': [0, 1, -1],
        'wdtype': [numpy.int32, numpy.float64],
        'output': [None, numpy.int32, numpy.float64],
        'filter': ['minimum_filter1d', 'maximum_filter1d']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'axis': [0, 1, -1],
        'mode': ['constant'],
        'cval': [-1.0, 0.0, 1.0],
        'origin': [0],
        'wdtype': [numpy.int32, numpy.float64],
        'output': [None],
        'filter': ['minimum_filter1d', 'maximum_filter1d']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'ksize': [3, 4],
        'axis': [0, 1, -1],
        'mode': ['nearest', 'mirror', 'wrap'],
        'cval': [0.0],
        'origin': [0],
        'wdtype': [numpy.int32, numpy.float64],
        'output': [None],
        'filter': ['minimum_filter1d', 'maximum_filter1d']
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestMinimumMaximum1DFilter(unittest.TestCase):
    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, axis=self.axis, output=self.output, mode=self.mode,
                      cval=self.cval, origin=self.origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve1d_and_correlate1d(self, xp, scp):
        a = testing.shaped_random(self.shape, xp, self.x_dtype)
        w = testing.shaped_random((self.ksize,), xp, self.x_dtype)
        return self._filter(xp, scp, a, w)

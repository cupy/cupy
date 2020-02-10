import unittest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


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
        if self.adtype == self.wdtype or self.adtype == self.output:
            return xp.array(True)
        a = testing.shaped_random(self.shape, xp, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, wdtype)
        return self._filter(xp, scp, a, w)


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

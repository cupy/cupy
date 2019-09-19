import unittest

import numpy

from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
    'ksize': [3, 4],
    'mode': ['reflect', 'constant', 'nearest', 'mirror', 'wrap'],
    'cval': [0.0],
    'origin': [-1, 0, 1],
    'output': [None, numpy.float32, numpy.float64],
    'filter': ['convolve', 'correlate']
}))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveAndCorrelate(unittest.TestCase):

    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, output=self.output, mode=self.mode,
                      cval=self.cval, origin=self.origin)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolve_float(self, xp, scp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, dtype)
        return self._filter(xp, scp, a, w)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_convolve_int(self, xp, scp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, dtype)
        return self._filter(xp, scp, a, w)

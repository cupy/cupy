import unittest

import numpy
import pytest

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
    'cval': [0.0, 1.0],
    'origin': [0, 1, None],
    'adtype': [numpy.int8, numpy.int16, numpy.int32,
               numpy.float32, numpy.float64],
    'wdtype': [None, numpy.int32, numpy.float64],
    'output': [None, numpy.int32, numpy.float64],
    'filter': ['convolve', 'correlate']
}))
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
            pytest.skip('skip duplicated test.')
        a = testing.shaped_random(self.shape, xp, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, wdtype)
        return self._filter(xp, scp, a, w)

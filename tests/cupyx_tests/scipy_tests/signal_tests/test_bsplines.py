import unittest

from cupy import testing

import cupyx.scipy.signal  # NOQA

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'input': [(256, 256), (4, 512), (512, 3)],
    'hrow': [1, 3, 4],
    'hcol': [1, 3, 4],
    'dtype': ['int32', 'float32', 'float64'],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSepFIR2d(unittest.TestCase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_sepfir2d(self, xp, scp):
        input = testing.shaped_random(self.input, xp, self.dtype)
        hrow = testing.shaped_random((self.hrow,), xp, self.dtype)
        hcol = testing.shaped_random((self.hcol,), xp, self.dtype)
        return scp.signal.sepfir2d(input, hrow, hcol)

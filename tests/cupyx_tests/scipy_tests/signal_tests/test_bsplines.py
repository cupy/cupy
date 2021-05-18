import sys
import unittest

from cupy import testing

import cupyx.scipy.signal  # NOQA

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'input': [(256, 256), (4, 512), (512, 3)],
    'hrow': [1, 3],
    'hcol': [1, 3],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSepFIR2d(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_sepfir2d(self, xp, scp, dtype):
        if sys.platform.startswith('win32') and xp.dtype(dtype).kind in 'c':
            # it's likely a SciPy bug (ValueError: incorrect type), so we
            # do this to effectively skip testing it
            return xp.zeros(10)

        input = testing.shaped_random(self.input, xp, dtype)
        hrow = testing.shaped_random((self.hrow,), xp, dtype)
        hcol = testing.shaped_random((self.hcol,), xp, dtype)
        return scp.signal.sepfir2d(input, hrow, hcol)

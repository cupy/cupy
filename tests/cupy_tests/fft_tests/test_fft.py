import unittest

import numpy as np

from cupy import testing


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'norm': [None, 'ortho'],
}))
@testing.gpu
class TestFft(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6, atol=1e-7)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

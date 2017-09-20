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
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7)
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(*testing.product({
    's1': [None, 5, 10, 15],
    's2': [None, 5, 10, 15],
    'norm': [None, 'ortho'],
}))
@testing.gpu
class TestFft2(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7)
    def test_fft2(self, xp, dtype):
        a = testing.shaped_random((10, 10), xp, dtype)
        out = xp.fft.fft2(a, s=(self.s1, self.s2), norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7)
    def test_ifft2(self, xp, dtype):
        a = testing.shaped_random((10, 10), xp, dtype)
        out = xp.fft.ifft2(a, s=(self.s1, self.s2), norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

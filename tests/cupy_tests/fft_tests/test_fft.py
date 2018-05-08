import unittest

import numpy as np

from cupy import testing


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho', ''],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft2(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftn(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft(self, xp, dtype):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.n is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfft2(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2(self, xp, dtype):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.s is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)
        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfftn(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestHfft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ihfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ihfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'n': 1, 'd': 1},
    {'n': 10, 'd': 0.5},
    {'n': 100, 'd': 2},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftfreq(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftfreq(self, xp, dtype):
        out = xp.fft.fftfreq(self.n, self.d)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfftfreq(self, xp, dtype):
        out = xp.fft.rfftfreq(self.n, self.d)

        return out


@testing.parameterize(
    {'shape': (5,), 'axes': None},
    {'shape': (5,), 'axes': 0},
    {'shape': (10,), 'axes': None},
    {'shape': (10,), 'axes': 0},
    {'shape': (10, 10), 'axes': None},
    {'shape': (10, 10), 'axes': 0},
    {'shape': (10, 10), 'axes': (0, 1)},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftshift(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftshift(x, self.axes)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftshift(x, self.axes)

        return out

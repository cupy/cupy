import unittest

from cupy import testing
import cupyx.scipy.fft as cp_fft
import numpy as np
import cupy as cp
import pytest


def _fft_module(xp):
    # Test cupyx.scipy against numpy since scipy.fft is not yet released
    if xp != np:
        return cp_fft
    else:
        return np.fft


def _correct_np_dtype(xp, dtype, out):
    # NumPy always transforms in double precision, cast output to correct type
    if xp == np:
        if dtype in [np.float16, np.float32, np.complex64]:
            if out.dtype.kind == 'f':
                return out.astype(np.float32)
            else:
                return out.astype(np.complex64)
    return out


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(9,), (10,), (10, 9), (10, 10)],
    'axis': [-1, 0],
    'norm': [None, 'ortho']
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).fft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).fft(x, n=self.n, axis=self.axis, norm=self.norm,
                                  **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4)],
        's': [None, (1, 5)],
        'axes': [None, (-2, -1), (-1, -2), (0,)],
        'norm': [None, 'ortho']
    })
    + testing.product({
        'shape': [(2, 3, 4)],
        's': [None, (1, 5), (1, 4, 10)],
        'axes': [None, (-2, -1), (-1, -2, -3)],
        'norm': [None, 'ortho']
    })))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft2(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes,
                                   norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).ifft2(
            x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).ifft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4)],
        's': [None, (1, 5)],
        'axes': [None, (-2, -1), (-1, -2), (0,)],
        'norm': [None, 'ortho']
    })
    + testing.product({
        'shape': [(2, 3, 4)],
        's': [None, (1, 5), (1, 4, 10)],
        'axes': [None, (-2, -1), (-1, -2, -3)],
        'norm': [None, 'ortho']
    })
    + testing.product({
        'shape': [(2, 3, 4, 5)],
        's': [None],
        'axes': [None, (0, 1, 2, 3)],
        'norm': [None, 'ortho']
    })))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftn(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes,
                                   norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes,
                                   norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(9,), (10,), (10, 9), (10, 10)],
    'axis': [-1, 0],
    'norm': [None, 'ortho']
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis,
                                   norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).irfft(x, n=self.n, axis=self.axis,
                                    norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).irfft(x, n=self.n, axis=self.axis,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


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
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).rfft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).rfft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).irfft2(x, s=self.s, axes=self.axes,
                                     norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).irfft2(x, s=self.s, axes=self.axes,
                                     norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


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
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).rfftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).rfftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).irfftn(x, s=self.s, axes=self.axes,
                                     norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).irfftn(x, s=self.s, axes=self.axes,
                                     norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'axis': [0, -1],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestHfft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_hfft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).hfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_hfft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).hfft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ihfft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).ihfft(x, n=self.n, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ihfft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp == np else {'overwrite_x': True}
        out = _fft_module(xp).ihfft(x, n=self.n, norm=self.norm,
                                    **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)


@testing.gpu
@pytest.mark.parametrize('func', [
    cp_fft.fft2, cp_fft.ifft2, cp_fft.rfft2, cp_fft.irfft2,
    cp_fft.fftn, cp_fft.ifftn, cp_fft.rfftn, cp_fft.irfftn])
def test_scalar_shape_axes(func):
    x = testing.shaped_random((10, 10), cp)
    y_scalar = func(x, s=5, axes=-1)
    y_normal = func(x, s=(5,), axes=(-1,))
    testing.assert_allclose(y_scalar, y_normal)

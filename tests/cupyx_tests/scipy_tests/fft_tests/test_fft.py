import unittest

import numpy as np
try:
    # scipy.fft is available since scipy v1.4.0+
    import scipy.fft as scipy_fft
except ImportError:
    scipy_fft = None
    scipy = None
else:
    import scipy
import pytest

import cupy as cp
from cupy import testing
from cupy.fft._fft import _default_fft_func, _fftn
import cupyx.scipy.fft as cp_fft
from cupyx.scipy.fft import _scipy_150


def _fft_module(xp):
    if xp is not np:
        return cp_fft
    else:
        if scipy_fft is not None:
            return scipy_fft
        else:  # fallback to numpy when scipy is unavailable
            return np.fft


def _correct_np_dtype(xp, dtype, out):
    # NumPy always transforms in double precision, cast output to correct type
    if xp is np and scipy_fft is None:
        if dtype in [np.float16, np.float32, np.complex64]:
            if out.dtype.kind == 'f':
                return out.astype(np.float32)
            else:
                return out.astype(np.complex64)
    return out


def _skip_forward_backward(norm):
    if norm in ('backward', 'forward'):
        if (scipy_fft is not None
                and not (np.lib.NumpyVersion(scipy.__version__) >= '1.6.0')):
            pytest.skip('forward/backward is supported by SciPy 1.6.0+')
        elif (scipy_fft is None
                and not (np.lib.NumpyVersion(np.__version__) >= '1.20.0')):
            pytest.skip('forward/backward is supported by NumPy 1.20+')


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(9,), (10,), (10, 9), (10, 10)],
    'axis': [-1, 0],
    'norm': [None, 'backward', 'ortho', 'forward', '']
}))
@testing.gpu
class TestFft(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).fft(x, n=self.n, axis=self.axis, norm=self.norm,
                                  **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis)}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).fft(x, n=self.n, axis=self.axis, norm=self.norm,
                                  **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis), 'overwrite_x': True}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).fft(x, n=self.n, axis=self.axis, norm=self.norm,
                                  **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(x, shape=self.n,
                                                axes=self.axis)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).fft(x, n=self.n, axis=self.axis)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).fft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.fft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @unittest.skipIf(scipy_fft is None or not _scipy_150,
                     'need scipy >= 1.5.0')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft_backend_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis)}
            backend = cp_fft
        else:
            # scipy raises NotImplementedError if plan is not None
            overwrite_kw = {'plan': None}
            backend = 'scipy'
        with scipy_fft.set_backend(backend):
            out = scipy_fft.fft(x, n=self.n, axis=self.axis, norm=self.norm,
                                **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis)}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis), 'overwrite_x': True}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(x, shape=self.n,
                                                axes=self.axis)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).ifft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ifft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @unittest.skipIf(scipy_fft is None or not _scipy_150,
                     'need scipy >= 1.5.0')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft_backend_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis)}
            backend = cp_fft
        else:
            # scipy raises NotImplementedError if plan is not None
            overwrite_kw = {'plan': None}
            backend = 'scipy'
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ifft(x, n=self.n, axis=self.axis, norm=self.norm,
                                 **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*(
    testing.product_dict(
        testing.product({
            'shape': [(3, 4)],
            's': [None, (1, 5)],
            'axes': [None, (-2, -1), (-1, -2), (0,)],
        })
        + testing.product({
            'shape': [(2, 3, 4)],
            's': [None, (1, 5), (1, 4, 10)],
            'axes': [None, (-2, -1), (-1, -2, -3)],
        }),
        testing.product({
            'norm': [None, 'backward', 'ortho', 'forward', '']
        })
    )
))
@testing.gpu
class TestFft2(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes,
                                   norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes, norm=self.norm,
                                   **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes), 'overwrite_x': True}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(x, shape=self.s,
                                                axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).fft2(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.fft2(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @unittest.skipIf(scipy_fft is None or not _scipy_150,
                     'need scipy >= 1.5.0')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2_backend_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
            backend = cp_fft
        else:
            # scipy raises NotImplementedError if plan is not None
            overwrite_kw = {'plan': None}
            backend = 'scipy'
        with scipy_fft.set_backend(backend):
            out = scipy_fft.fft2(x, s=self.s, axes=self.axes, norm=self.norm,
                                 **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).ifft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).ifft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes), 'overwrite_x': True}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).ifft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(x, shape=self.s,
                                                axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).ifft2(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).ifft2(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ifft2(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @unittest.skipIf(scipy_fft is None or not _scipy_150,
                     'need scipy >= 1.5.0')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2_backend_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
            backend = cp_fft
        else:
            # scipy raises NotImplementedError if plan is not None
            overwrite_kw = {'plan': None}
            backend = 'scipy'
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ifft2(x, s=self.s, axes=self.axes, norm=self.norm,
                                  **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*(
    testing.product_dict(
        testing.product({
            'shape': [(3, 4)],
            's': [None, (1, 5)],
            'axes': [None, (-2, -1), (-1, -2), (0,)],
        })
        + testing.product({
            'shape': [(2, 3, 4)],
            's': [None, (1, 5), (1, 4, 10)],
            'axes': [None, (0, 1), (-2, -1), (-1, -2, -3)],
        })
        + testing.product({
            'shape': [(2, 3, 4, 5)],
            's': [None],
            'axes': [None, (0, 1, 2, 3)],
        }),
        testing.product({
            'norm': [None, 'backward', 'ortho', 'forward', '']
        })
    )
))
@testing.gpu
class TestFftn(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes,
                                   norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes, norm=self.norm,
                                   **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes), 'overwrite_x': True}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(x, shape=self.s,
                                                axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).fftn(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.fftn(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @unittest.skipIf(scipy_fft is None or not _scipy_150,
                     'need scipy >= 1.5.0')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_backend_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
            backend = cp_fft
        else:
            # scipy raises NotImplementedError if plan is not None
            overwrite_kw = {'plan': None}
            backend = 'scipy'
        with scipy_fft.set_backend(backend):
            out = scipy_fft.fftn(x, s=self.s, axes=self.axes, norm=self.norm,
                                 **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes), 'overwrite_x': True}
        else:
            overwrite_kw = {}
        out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(x, shape=self.s,
                                                axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).ifftn(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ifftn(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @unittest.skipIf(scipy_fft is None or not _scipy_150,
                     'need scipy >= 1.5.0')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_backend_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_fft_func(x, s=self.s, axes=self.axes) is not _fftn:
            return x
        x_orig = x.copy()
        if xp is cp:
            overwrite_kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.s, axes=self.axes)}
            backend = cp_fft
        else:
            # scipy raises NotImplementedError if plan is not None
            overwrite_kw = {'plan': None}
            backend = 'scipy'
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ifftn(x, s=self.s, axes=self.axes, norm=self.norm,
                                  **overwrite_kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(9,), (10,), (10, 9), (10, 10)],
    'axis': [-1, 0],
    'norm': [None, 'backward', 'ortho', 'forward', '']
}))
@testing.gpu
class TestRfft(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis,
                                   norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        if xp is cp:
            kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis, value_type='R2C')}
        else:
            kw = {}
        out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.rfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if xp is cp:
            kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis, value_type='R2C'),
                'overwrite_x': True}
        else:
            kw = {}
        out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis, value_type='R2C')
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).rfft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    # the irfft tests show a slightly different results in CUDA 11.0 when
    # compared to SciPy 1.6.1
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).irfft(x, n=self.n, axis=self.axis,
                                    norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).irfft(x, n=self.n, axis=self.axis,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        if xp is cp:
            kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis, value_type='C2R')}
        else:
            kw = {}
        out = _fft_module(xp).irfft(
            x, n=self.n, axis=self.axis, norm=self.norm, **kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if xp is cp:
            kw = {'plan': _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis, value_type='C2R'),
                'overwrite_x': True}
        else:
            kw = {}
        out = _fft_module(xp).irfft(
            x, n=self.n, axis=self.axis, norm=self.norm, **kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            plan = _fft_module(xp).get_fft_plan(
                x, shape=self.n, axes=self.axis, value_type='C2R')
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).irfft(x, n=self.n, axis=self.axis)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).irfft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.irfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)


def _skip_hipFFT_PlanNd_bug(axes, shape):
    if cp.cuda.runtime.is_hip:
        # TODO(leofang): test newer ROCm versions
        if (axes == (0, 1) and shape == (2, 3, 4)):
            raise unittest.SkipTest("hipFFT's PlanNd for this case is buggy, "
                                    "so Plan1d is generated instead")


@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestRfft2(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).rfft2(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2_plan(self, xp, dtype):
        _skip_hipFFT_PlanNd_bug(self.axes, self.shape)
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='R2C')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan}
        else:
            kw = {}
        out = _fft_module(xp).rfft2(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2_overwrite_plan(self, xp, dtype):
        _skip_hipFFT_PlanNd_bug(self.axes, self.shape)
        x = testing.shaped_random(self.shape, xp, dtype)

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='R2C')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan, 'overwrite_x': True}
        else:
            kw = {}
        out = _fft_module(xp).rfft2(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2_plan_manager(self, xp, dtype):
        _skip_hipFFT_PlanNd_bug(self.axes, self.shape)
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='R2C')
        except ValueError:
            return x

        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).rfft2(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).rfft2(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.rfft2(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70 and
                        10020 >= cp.cuda.runtime.runtimeGetVersion() >= 10010,
                        reason="Known to fail with Pascal or older")
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

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70 and
                        10020 >= cp.cuda.runtime.runtimeGetVersion() >= 10010,
                        reason="Known to fail with Pascal or older")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).irfft2(x, s=self.s, axes=self.axes,
                                     norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70,
                        reason="Known to fail with Pascal or older")
    @unittest.skipIf(cp.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='C2R')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan}
        else:
            kw = {}
        out = _fft_module(xp).irfft2(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        testing.assert_array_equal(x, x_orig)

        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70,
                        reason="Known to fail with Pascal or older")
    @unittest.skipIf(cp.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='C2R')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan, 'overwrite_x': True}
        else:
            kw = {}
        out = _fft_module(xp).irfft2(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70,
                        reason="Known to fail with Pascal or older")
    @unittest.skipIf(cp.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='C2R')
        except ValueError:
            return x

        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).irfft2(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).irfft2(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)

        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70 and
                        10020 >= cp.cuda.runtime.runtimeGetVersion() >= 10010,
                        reason="Known to fail with Pascal or older")
    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.irfft2(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestRfftn(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).rfftn(x, s=self.s, axes=self.axes,
                                    norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_plan(self, xp, dtype):
        _skip_hipFFT_PlanNd_bug(self.axes, self.shape)
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='R2C')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan}
        else:
            kw = {}
        out = _fft_module(xp).rfftn(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_overwrite_plan(self, xp, dtype):
        _skip_hipFFT_PlanNd_bug(self.axes, self.shape)
        x = testing.shaped_random(self.shape, xp, dtype)

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='R2C')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan, 'overwrite_x': True}
        else:
            kw = {}
        out = _fft_module(xp).rfftn(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_plan_manager(self, xp, dtype):
        _skip_hipFFT_PlanNd_bug(self.axes, self.shape)
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='R2C')
        except ValueError:
            return x

        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).rfftn(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).rfftn(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.rfftn(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70 and
                        10020 >= cp.cuda.runtime.runtimeGetVersion() >= 10010,
                        reason="Known to fail with Pascal or older")
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

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70 and
                        10020 >= cp.cuda.runtime.runtimeGetVersion() >= 10010,
                        reason="Known to fail with Pascal or older")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).irfftn(x, s=self.s, axes=self.axes,
                                     norm=self.norm, **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70,
                        reason="Known to fail with Pascal or older")
    @unittest.skipIf(cp.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='C2R')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan}
        else:
            kw = {}
        out = _fft_module(xp).irfftn(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        testing.assert_array_equal(x, x_orig)

        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70,
                        reason="Known to fail with Pascal or older")
    @unittest.skipIf(cp.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_overwrite_plan(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='C2R')
        except ValueError:
            return x

        if xp is cp:
            kw = {'plan': plan, 'overwrite_x': True}
        else:
            kw = {}
        out = _fft_module(xp).irfftn(
            x, s=self.s, axes=self.axes, norm=self.norm, **kw)
        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70,
                        reason="Known to fail with Pascal or older")
    @unittest.skipIf(cp.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_plan_manager(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()

        # hack: skip testing if getting a cuFFT plan is impossible
        try:
            plan = _fft_module(cp).get_fft_plan(
                x, shape=self.s, axes=self.axes, value_type='C2R')
        except ValueError:
            return x

        if xp is cp:
            from cupy.cuda.cufft import get_current_plan
            with plan:
                assert id(plan) == id(get_current_plan())
                out = _fft_module(xp).irfftn(x, s=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:
            out = _fft_module(xp).irfftn(x, s=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)

        return _correct_np_dtype(xp, dtype, out)

    @pytest.mark.skipif(int(cp.cuda.device.get_compute_capability()) < 70 and
                        10020 >= cp.cuda.runtime.runtimeGetVersion() >= 10010,
                        reason="Known to fail with Pascal or older")
    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.irfftn(x, s=self.s, axes=self.axes, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'axis': [0, -1],
    'norm': [None, 'backward', 'ortho', 'forward', ''],
}))
@testing.gpu
class TestHfft(unittest.TestCase):

    def setUp(self):
        _skip_forward_backward(self.norm)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=4e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_hfft(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = _fft_module(xp).hfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
        return _correct_np_dtype(xp, dtype, out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=4e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_hfft_overwrite(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).hfft(x, n=self.n, axis=self.axis, norm=self.norm,
                                   **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    # TODO(leofang): rewrite this test when we support R2C/C2R cuFFT plans
    @testing.for_all_dtypes()
    def test_hfft_plan(self, dtype):
        x = testing.shaped_random(self.shape, cp, dtype)
        with pytest.raises(NotImplementedError, match='not yet supported'):
            _fft_module(cp).hfft(x, n=self.n, axis=self.axis,
                                 norm=self.norm, plan='abc')

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=4e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_hfft_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.hfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
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
        overwrite_kw = {} if xp is np else {'overwrite_x': True}
        out = _fft_module(xp).ihfft(x, n=self.n, norm=self.norm,
                                    **overwrite_kw)
        return _correct_np_dtype(xp, dtype, out)

    # TODO(leofang): rewrite this test when we support R2C/C2R cuFFT plans
    @testing.for_all_dtypes(no_complex=True)
    def test_ihfft_plan(self, dtype):
        x = testing.shaped_random(self.shape, cp, dtype)
        with pytest.raises(NotImplementedError, match='not yet supported'):
            _fft_module(cp).ihfft(x, n=self.n, axis=self.axis,
                                  norm=self.norm, plan='abc')

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ihfft_backend(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            out = scipy_fft.ihfft(x, n=self.n, axis=self.axis, norm=self.norm)
        testing.assert_array_equal(x, x_orig)
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

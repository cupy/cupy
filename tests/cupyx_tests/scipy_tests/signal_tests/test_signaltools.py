import unittest

import pytest

import numpy as np

import cupy
from cupy.cuda import runtime
from cupy import testing

import cupyx.scipy.signal

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'size1': [(10,), (5, 10), (10, 3), (3, 4, 10)],
    'size2': [3, 4, 5, 10],
    'mode': ['full', 'same', 'valid'],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveCorrelate(unittest.TestCase):
    def _filter(self, func, dtype, xp, scp):
        in1 = testing.shaped_random(self.size1, xp, dtype)
        in2 = testing.shaped_random((self.size2,)*in1.ndim, xp, dtype)
        return getattr(scp.signal, func)(in1, in2, self.mode, method='direct')

    tols = {np.float32: 1e-5, np.complex64: 1e-5,
            np.float16: 1e-3, 'default': 1e-10}

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve(self, xp, scp, dtype):
        return self._filter('convolve', dtype, xp, scp)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate(self, xp, scp, dtype):
        return self._filter('correlate', dtype, xp, scp)


@testing.parameterize(*testing.product({
    'size1': [(10,), (5, 10), (10, 3), (3, 4, 10)],
    'size2': [3, 4, 5, 10],
    'mode': ['full', 'same', 'valid'],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestFFTConvolve(unittest.TestCase):
    def _filter(self, func, dtype, xp, scp, **kwargs):
        in1 = testing.shaped_random(self.size1, xp, dtype)
        in2 = testing.shaped_random((self.size2,)*in1.ndim, xp, dtype)
        return getattr(scp.signal, func)(in1, in2, self.mode, **kwargs)

    tols = {np.float32: 1e-3, np.complex64: 1e-3,
            np.float16: 1e-3, 'default': 1e-8}

    def _hip_skip_invalid_condition(self):
        invalid_condition = [
            ('full', 4), ('full', 5), ('full', 10),
            ('same', 3), ('same', 4), ('same', 5), ('same', 10),
            ('valid', 10)]
        if (runtime.is_hip and self.size1 == (3, 4, 10)
                and (self.mode, self.size2) in invalid_condition):
            pytest.xfail('ROCm/HIP may have a bug')

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_fftconvolve(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()
        return self._filter('fftconvolve', dtype, xp, scp)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve_fft(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()
        return self._filter('convolve', dtype, xp, scp, method='fft')

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate_fft(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()
        return self._filter('correlate', dtype, xp, scp, method='fft')


@testing.parameterize(*testing.product({
    'size1': [(10,), (5, 10), (10, 3), (3, 10, 15)],
    'size2': [3, 4, 5, 10, None],
    'mode': ['full', 'same', 'valid'],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestOAConvolve(unittest.TestCase):
    tols = {np.float32: 1e-3, np.complex64: 1e-3,
            np.float16: 1e-3, 'default': 1e-8}

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_oaconvolve(self, xp, scp, dtype):
        if runtime.is_hip and self.size2 in [5, None]:
            pytest.xfail('ROCm/HIP may have a bug')
        in1 = testing.shaped_random(self.size1, xp, dtype)
        shape2 = self.size1 if self.size2 is None else (self.size2,)*in1.ndim
        in2 = testing.shaped_random(shape2, xp, dtype)
        return scp.signal.oaconvolve(in1, in2, self.mode)


@testing.parameterize(*(testing.product({
    'size1': [(5, 10), (10, 7)],
    'size2': [(3, 2), (3, 3), (2, 2), (10, 10), (11, 11)],
    'mode': ['full', 'same', 'valid'],
    'boundary': ['fill'],
    'fillvalue': [0, 1, -1],
}) + testing.product({
    'size1': [(5, 10), (10, 7)],
    'size2': [(3, 2), (3, 3), (2, 2), (10, 10), (11, 11)],
    'mode': ['full', 'same', 'valid'],
    'boundary': ['wrap', 'symm'],
    'fillvalue': [0],
})))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveCorrelate2D(unittest.TestCase):
    def _filter(self, func, dtype, xp, scp):
        if self.mode == 'full' and self.boundary != 'fill':
            # See https://github.com/scipy/scipy/issues/12685
            raise unittest.SkipTest('broken in scipy')
        in1 = testing.shaped_random(self.size1, xp, dtype)
        in2 = testing.shaped_random(self.size2, xp, dtype)
        return getattr(scp.signal, func)(in1, in2, self.mode, self.boundary,
                                         self.fillvalue)

    tols = {np.float32: 5e-4, np.complex64: 5e-4,
            np.float16: 1e-3, 'default': 1e-10}

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve2d(self, xp, scp, dtype):
        return self._filter('convolve2d', dtype, xp, scp)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate2d(self, xp, scp, dtype):
        return self._filter('correlate2d', dtype, xp, scp)


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full']
}))
class TestChooseConvMethod(unittest.TestCase):

    @testing.for_dtypes('efdFD')
    def test_choose_conv_method1(self, dtype):
        a = testing.shaped_arange((10000,), cupy, dtype)
        b = testing.shaped_arange((5000,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'fft'

    @testing.for_dtypes('efdFD')
    def test_choose_conv_method2(self, dtype):
        a = testing.shaped_arange((5000,), cupy, dtype)
        b = testing.shaped_arange((10000,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'fft'

    @testing.for_int_dtypes()
    def test_choose_conv_method_int(self, dtype):
        a = testing.shaped_arange((10,), cupy, dtype)
        b = testing.shaped_arange((5,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'direct'

    @testing.for_all_dtypes()
    def test_choose_conv_method_ndim(self, dtype):
        a = testing.shaped_arange((3, 4, 5), cupy, dtype)
        b = testing.shaped_arange((1, 2), cupy, dtype)
        with pytest.raises(NotImplementedError):
            cupyx.scipy.signal.choose_conv_method(a, b, mode=self.mode)

    @testing.for_all_dtypes()
    def test_choose_conv_method_zero_dim(self, dtype):
        a = testing.shaped_arange((), cupy, dtype)
        b = testing.shaped_arange((5,), cupy, dtype)
        with pytest.raises(NotImplementedError):
            cupyx.scipy.signal.choose_conv_method(a, b, mode=self.mode)


@testing.parameterize(*testing.product({
    'im': [(10,), (5, 10), (10, 3), (3, 4, 10)],
    'mysize': [3, 4, (3, 4, 5)],
    'noise': [False, True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestWiener(unittest.TestCase):
    tols = {np.float32: 1e-5, np.complex64: 1e-5,
            np.float16: 1e-3, 'default': 1e-10}

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=tols, rtol=tols, scipy_name='scp')
    def test_wiener(self, xp, scp, dtype):
        im = testing.shaped_random(self.im, xp, dtype)
        mysize = self.mysize
        if isinstance(mysize, tuple):
            mysize = mysize[:im.ndim]
        noise = (testing.shaped_random(self.im, xp, dtype)
                 if self.noise else None)
        out = scp.signal.wiener(im, mysize, noise)
        # Always returns float64 or complex128 data  in both scipy and
        # cupyx.scipy. Per-datatype tolerances are based on the output
        # data type but quality is based on input data type (if floating point)
        assert out.dtype == (np.complex128 if out.dtype.kind == 'c' else
                             np.float64)
        return out.astype(dtype, copy=False) if dtype in self.tols else out


@testing.parameterize(*testing.product({
    'a': [(10,), (5, 10), (10, 3), (3, 4, 10)],
    'domain': [3, 4, (3, 3, 5)],
    'rank': [0, 1, 2],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestOrderFilter(unittest.TestCase):
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-8, rtol=1e-8, scipy_name='scp',
                                 accept_error=ValueError)  # for even kernels
    def test_order_filter(self, xp, scp, dtype):
        a = testing.shaped_random(self.a, xp, dtype)
        d = self.domain
        d = d[:a.ndim] if isinstance(d, tuple) else (d,)*a.ndim
        domain = testing.shaped_random(d, xp) > 0.25
        rank = min(self.rank, domain.sum())
        return scp.signal.order_filter(a, domain, rank)


@testing.parameterize(*testing.product({
    'volume': [(10,), (5, 10), (10, 5), (5, 6, 10)],
    'kernel_size': [3, 4, (3, 3, 5)],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMedFilt(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-8, rtol=1e-8, scipy_name='scp',
                                 accept_error=ValueError)  # for even kernels
    def test_medfilt(self, xp, scp, dtype):
        volume = testing.shaped_random(self.volume, xp, dtype)
        kernel_size = self.kernel_size
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[:volume.ndim]
        return scp.signal.medfilt(volume, kernel_size)


@testing.parameterize(*testing.product({
    'input': [(5, 10), (10, 5)],
    'kernel_size': [3, 4, (3, 5)],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMedFilt2d(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-8, rtol=1e-8, scipy_name='scp',
                                 accept_error=ValueError)  # for even kernels
    def test_medfilt2d(self, xp, scp, dtype):
        input = testing.shaped_random(self.input, xp, dtype)
        kernel_size = self.kernel_size
        return scp.signal.medfilt2d(input, kernel_size)

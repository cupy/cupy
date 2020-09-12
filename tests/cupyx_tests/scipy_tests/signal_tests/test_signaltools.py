import unittest

import pytest

import cupy
from cupy import testing

import cupyx.scipy.signal

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'in1': [(10,), (5, 10), (10, 3), (3, 4, 10)],
    'in2': [3, 4, 5, 10],
    'mode': ['full', 'same', 'valid'],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveCorrelate(unittest.TestCase):
    def _filter(self, func, dtype, xp, scp):
        in1 = testing.shaped_random(self.in1, xp, dtype)
        in2 = testing.shaped_random((self.in2,)*in1.ndim, xp, dtype)
        return getattr(scp.signal, func)(in1, in2, self.mode, method='direct')

    # TODO: support complex
    # Note: float16 is tested separately
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve(self, xp, scp, dtype):
        return self._filter('convolve', dtype, xp, scp)

    # TODO: support complex
    # Note: float16 is tested separately
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate(self, xp, scp, dtype):
        return self._filter('correlate', dtype, xp, scp)

    # float16 has significantly worse error tolerances
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve_float16(self, xp, scp, dtype=cupy.float16):
        return self._filter('convolve', dtype, xp, scp)

    # float16 has significantly worse error tolerances
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate_float16(self, xp, scp, dtype=cupy.float16):
        return self._filter('correlate', dtype, xp, scp)


@testing.parameterize(*(testing.product({
    'in1': [(5, 10), (10, 7)],
    'in2': [(3, 2), (3, 3), (2, 2), (10, 10), (11, 11)],
    'mode': ['full', 'same', 'valid'],
    'boundary': ['fill'],
    'fillvalue': [0, 1, -1],
}) + testing.product({
    'in1': [(5, 10), (10, 7)],
    'in2': [(3, 2), (3, 3), (2, 2), (10, 10), (11, 11)],
    'mode': ['full', 'same', 'valid'],
    'boundary': ['wrap', 'symm'],
    'fillvalue': [0],
})))
@testing.gpu
@testing.with_requires('scipy')
class TestConvolveCorrelate2D(unittest.TestCase):
    def _filter(self, func, dtype, xp, scp):
        if self.mode == 'full' and self.boundary != 'constant':
            # See https://github.com/scipy/scipy/issues/12685
            raise unittest.SkipTest('broken in scipy')
        in1 = testing.shaped_random(self.in1, xp, dtype)
        in2 = testing.shaped_random(self.in2, xp, dtype)
        return getattr(scp.signal, func)(in1, in2, self.mode, self.boundary,
                                         self.fillvalue)

    # TODO: support complex
    # Note: float16 is tested separately
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve2d(self, xp, scp, dtype):
        return self._filter('convolve2d', dtype, xp, scp)

    # TODO: support complex
    # Note: float16 is tested separately
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate2d(self, xp, scp, dtype):
        return self._filter('correlate2d', dtype, xp, scp)

    # float16 has significantly worse error tolerances
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3, scipy_name='scp',
                                 accept_error=ValueError)
    def test_convolve2d_float16(self, xp, scp, dtype=cupy.float16):
        return self._filter('convolve2d', dtype, xp, scp)

    # float16 has significantly worse error tolerances
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3, scipy_name='scp',
                                 accept_error=ValueError)
    def test_correlate2d_float16(self, xp, scp, dtype=cupy.float16):
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
    # TODO: support complex
    # Note: float16 is tested separately
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_wiener(self, xp, scp, dtype):
        im = testing.shaped_random(self.im, xp, dtype)
        mysize = self.mysize
        if isinstance(mysize, tuple):
            mysize = mysize[:im.ndim]
        noise = (testing.shaped_random(self.im, xp, dtype)
                 if self.noise else None)
        return scp.signal.wiener(im, mysize, noise)

    # float16 has significantly worse error tolerances
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3, scipy_name='scp')
    def test_wiener_float16(self, xp, scp, dtype=cupy.float16):
        im = testing.shaped_random(self.im, xp, dtype)
        mysize = self.mysize
        if isinstance(mysize, tuple):
            mysize = mysize[:im.ndim]
        noise = (testing.shaped_random(self.im, xp, dtype)
                 if self.noise else None)
        return scp.signal.wiener(im, mysize, noise)


@testing.parameterize(*testing.product({
    'a': [(10,), (5, 10), (10, 3), (3, 4, 10)],
    'domain': [3, 4, (3, 3, 5)],
    'rank': [0, 1, 2],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestOrderFilter(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
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
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
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
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)  # for even kernels
    def test_medfilt2d(self, xp, scp, dtype):
        input = testing.shaped_random(self.input, xp, dtype)
        kernel_size = self.kernel_size
        return scp.signal.medfilt2d(input, kernel_size)

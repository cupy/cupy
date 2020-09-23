import unittest

import numpy

from cupy import testing
# TODO (grlee77): use fft instead of fftpack once min. supported scipy >= 1.4
import cupyx.scipy.fft  # NOQA
import cupyx.scipy.fftpack  # NOQA
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.fft  # NOQA
    import scipy.fftpack  # NOQA
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@testing.parameterize(
    *(
        testing.product(
            {
                "shape": [(32, 16), (31, 15)],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "shift": [1, -3, (5, 5.3), (3, 5)],
            }
        )
        + testing.product(
            {
                "shape": [(5, 16, 7), ],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "shift": [3, (-1, 2.5, 1)],
            }
        )
        + testing.product(
            {
                "shape": [(15, ), ],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "shift": [8.5, (5,)],
            }
        )
    )
)
@testing.gpu
@testing.with_requires("scipy")
class TestFourierShift(unittest.TestCase):

    def _test_real_nd(self, xp, scp, x, real_axis):
        a = scp.fft.rfft(x, axis=real_axis)
        # complex-valued FFTs on all other axes
        complex_axes = tuple([ax for ax in range(x.ndim) if ax != real_axis])
        if complex_axes:
            a = scp.fft.fftn(a, axes=complex_axes)

        a = scp.ndimage.fourier_shift(
            a, self.shift, n=x.shape[real_axis], axis=real_axis)

        if complex_axes:
            a = scp.fft.ifftn(a, axes=complex_axes)
        a = scp.fft.irfft(a, axis=real_axis)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.with_requires("scipy>=1.4.0")
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_real_fft_axis0(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        if x.dtype.kind == 'c':
            # skip: can't use rfft on complex-valued x
            return x
        return self._test_real_nd(xp, scp, x, 0)

    @testing.with_requires("scipy>=1.4.0")
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_real_fft_axis1(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        if x.dtype.kind == 'c' or x.ndim < 2:
            # skip: can't use rfft along axis 1 on complex-valued x or 1d x
            return x
        return self._test_real_nd(xp, scp, x, 1)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_complex_fft(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        a = scp.fftpack.fftn(x)
        a = scp.ndimage.fourier_shift(a, self.shift)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_complex_fft_with_output(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        a = scp.fftpack.fftn(x)
        scp.ndimage.fourier_shift(a, self.shift, output=a)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)


@testing.parameterize(
    *(
        testing.product(
            {
                "shape": [(32, 16), (31, 15)],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "sigma": [1, (5, 5.3), (3, 5)],
            }
        )
        + testing.product(
            {
                "shape": [(5, 16, 7), ],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "sigma": [3, (1, 2.5, 3)],
            }
        )
        + testing.product(
            {
                "shape": [(15, ), ],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "sigma": [8.5, (5,)],
            }
        )
    )
)
@testing.gpu
@testing.with_requires("scipy")
class TestFourierGaussian(unittest.TestCase):

    def _test_real_nd(self, xp, scp, x, real_axis):
        a = scp.fft.rfft(x, axis=real_axis)
        # complex-valued FFTs on all other axes
        complex_axes = tuple([ax for ax in range(x.ndim) if ax != real_axis])
        if complex_axes:
            a = scp.fft.fftn(a, axes=complex_axes)

        a = scp.ndimage.fourier_gaussian(
            a, self.sigma, n=x.shape[real_axis], axis=real_axis)

        if complex_axes:
            a = scp.fft.ifftn(a, axes=complex_axes)
        a = scp.fft.irfft(a, axis=real_axis)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.with_requires("scipy>=1.4.0")
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_real_fft_axis0(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        if x.dtype.kind == 'c':
            # skip: can't use rfft on complex-valued x
            return x
        return self._test_real_nd(xp, scp, x, 0)

    @testing.with_requires("scipy>=1.4.0")
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_real_fft_axis1(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        if x.dtype.kind == 'c' or x.ndim < 2:
            # skip: can't use rfft along axis 1 on complex-valued x or 1d x
            return x
        return self._test_real_nd(xp, scp, x, 1)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_complex_fft(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        a = scp.fftpack.fftn(x)
        a = scp.ndimage.fourier_gaussian(a, self.sigma)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_complex_fft_with_output(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        a = scp.fftpack.fftn(x)
        scp.ndimage.fourier_gaussian(a, self.sigma, output=a)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)


@testing.parameterize(
    *(
        testing.product(
            {
                "shape": [(32, 16), (31, 15)],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "size": [1, (5, 5.3), (3, 5)],
            }
        )
        + testing.product(
            {
                "shape": [(5, 16, 7), ],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "size": [3, (1, 2.5, 3)],
            }
        )
        + testing.product(
            {
                "shape": [(15, ), ],
                "dtype": [numpy.float32, numpy.float64, numpy.complex64,
                          numpy.complex128],
                "size": [8.5, (5,)],
            }
        )
    )
)
@testing.gpu
@testing.with_requires("scipy")
class TestFourierUniform(unittest.TestCase):

    def _test_real_nd(self, xp, scp, x, real_axis):
        a = scp.fft.rfft(x, axis=real_axis)
        # complex-valued FFTs on all other axes
        complex_axes = tuple([ax for ax in range(x.ndim) if ax != real_axis])
        if complex_axes:
            a = scp.fft.fftn(a, axes=complex_axes)

        a = scp.ndimage.fourier_uniform(
            a, self.size, n=x.shape[real_axis], axis=real_axis)

        if complex_axes:
            a = scp.fft.ifftn(a, axes=complex_axes)
        a = scp.fft.irfft(a, axis=real_axis)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.with_requires("scipy>=1.4.0")
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_real_fft_axis0(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        if x.dtype.kind == 'c':
            # skip: can't use rfft on complex-valued x
            return x
        return self._test_real_nd(xp, scp, x, 0)

    @testing.with_requires("scipy>=1.4.0")
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_real_fft_axis1(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        if x.dtype.kind == 'c' or x.ndim < 2:
            # skip: can't use rfft along axis 1 on complex-valued x or 1d x
            return x
        return self._test_real_nd(xp, scp, x, 1)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_complex_fft(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        a = scp.fftpack.fftn(x)
        a = scp.ndimage.fourier_uniform(a, self.size)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_complex_fft_with_output(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.dtype)
        a = scp.fftpack.fftn(x)
        scp.ndimage.fourier_uniform(a, self.size, output=a)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

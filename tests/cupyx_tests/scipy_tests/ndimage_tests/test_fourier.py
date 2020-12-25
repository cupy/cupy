import cupy
import numpy
import pytest

from cupy import testing
# TODO (grlee77): use fft instead of fftpack once min. supported scipy >= 1.4
import cupyx.scipy.fft  # NOQA
import cupyx.scipy.fftpack  # NOQA
import cupyx.scipy.ndimage  # NOQA

try:
    # scipy.fft only available since SciPy 1.4.0
    import scipy.fft  # NOQA
except ImportError:
    pass

try:
    # These modules will be present in all supported SciPy versions
    import scipy
    import scipy.fftpack  # NOQA
    import scipy.ndimage  # NOQA
    scipy_version = numpy.lib.NumpyVersion(scipy.__version__)
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
class TestFourierShift:

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
class TestFourierGaussian:

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
class TestFourierUniform:

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


@testing.parameterize(
    *(
        testing.product(
            {
                'shape': [(32, 16), (31, 15)],
                'size': [1, (5, 5), (3, 5)],
            }
        )
        + testing.product(
            {
                'shape': [(5, 16, 7)],
                'size': [3, (1, 2, 4)],
            }
        )
        + testing.product(
            {
                'shape': [(15, ), ],
                'size': [8, (5,)],
            }
        )
    )
)
@testing.gpu
@testing.with_requires('scipy')
class TestFourierEllipsoid():
    def _test_real_nd(self, xp, scp, x, real_axis):
        if x.ndim == 1 and scipy_version < '1.5.3':
            # 1D case gives an incorrect result in SciPy < 1.5.3
            pytest.skip('scipy version to old')

        a = scp.fft.rfft(x, axis=real_axis)
        # complex-valued FFTs on all other axes
        complex_axes = tuple([ax for ax in range(x.ndim) if ax != real_axis])
        if complex_axes:
            a = scp.fft.fftn(a, axes=complex_axes)

        a = scp.ndimage.fourier_ellipsoid(
            a, self.size, n=x.shape[real_axis], axis=real_axis
        )

        if complex_axes:
            a = scp.fft.ifftn(a, axes=complex_axes)
        a = scp.fft.irfft(a, axis=real_axis)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_real_fft_axis0(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return self._test_real_nd(xp, scp, x, 0)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_real_fft_axis1(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if x.ndim < 2:
            # skip: there is no axis=1 on 1d arrays
            return x
        return self._test_real_nd(xp, scp, x, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_complex_fft(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if x.ndim == 1 and scipy_version < '1.5.3':
            # 1D case gives an incorrect result in SciPy < 1.5.3
            pytest.skip('scipy version to old')
        a = scp.fftpack.fftn(x)
        a = scp.ndimage.fourier_ellipsoid(a, self.size)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_complex_fft_with_output(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if x.ndim == 1 and scipy_version < '1.5.3':
            # 1D case gives an incorrect result in SciPy < 1.5.3
            pytest.skip('scipy version to old')
        a = scp.fftpack.fftn(x)
        scp.ndimage.fourier_ellipsoid(a.copy(), self.size, output=a)
        a = scp.fftpack.ifftn(a)
        if not x.dtype.kind == 'c':
            a = a.real
        return xp.ascontiguousarray(a)


@testing.gpu
@testing.with_requires('scipy')
class TestFourierEllipsoidInvalid():

    # SciPy < 1.5 raises ValueError instead of AxisError
    @testing.with_requires('scipy>=1.5.0')
    def test_0d_input(self):
        for xp, scp in zip((numpy, cupy), (scipy, cupyx.scipy)):
            with pytest.raises(numpy.AxisError):
                scp.ndimage.fourier_ellipsoid(xp.asarray(5.0), size=2)
        return

    def test_4d_input(self):
        # SciPy should raise here too because >3d isn't implemented, but
        # as of 1.5.4, it does not.
        shape = (4, 6, 8, 10)
        for xp, scp in zip((cupy,), (cupyx.scipy,)):
            with pytest.raises(NotImplementedError):
                scp.ndimage.fourier_ellipsoid(xp.ones(shape), size=2)
        return

    # SciPy < 1.5 raises ValueError instead of AxisError
    @testing.with_requires('scipy>=1.5.0')
    def test_invalid_axis(self):
        # SciPy should raise here too because >3d isn't implemented, but
        # as of 1.5.4, it does not.
        shape = (6, 8)
        for xp, scp in zip((numpy, cupy), (scipy, cupyx.scipy)):
            with pytest.raises(numpy.AxisError):
                scp.ndimage.fourier_ellipsoid(xp.ones(shape), 2, axis=2)
            with pytest.raises(numpy.AxisError):
                scp.ndimage.fourier_ellipsoid(xp.ones(shape), 2, axis=-3)
        return

    def test_invalid_size(self):
        # test size length mismatch
        shape = (6, 8)
        for xp, scp in zip((numpy, cupy), (scipy, cupyx.scipy)):
            with pytest.raises(RuntimeError):
                scp.ndimage.fourier_ellipsoid(xp.ones(shape), size=(2, 3, 4))
            with pytest.raises(RuntimeError):
                scp.ndimage.fourier_ellipsoid(xp.ones(shape), size=(4,))
        return

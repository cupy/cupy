
import pytest

import numpy as np

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing
import cupyx.scipy.signal  # NOQA

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@pytest.mark.xfail(
    runtime.is_hip and driver.get_build_version() < 5_00_00000,
    reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestLombscargle:
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_frequency(self, dtype, xp, scp):
        """Test if frequency locations of peak corresponds to frequency of
        generated input signal.
        """
        dtype = xp.dtype(dtype)

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * xp.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        r = testing.shaped_random((nin,), xp, scale=1.0,
                                  dtype=dtype, seed=2353425)
        t = xp.linspace(0.01 * xp.pi, 10. * xp.pi, nin, dtype=dtype)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout, dtype=dtype)

        # Calculate Lomb-Scargle periodogram
        P = scp.signal.lombscargle(t, x, f)
        return P

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_amplitude(self, dtype, xp, scp):
        # Test if height of peak in normalized Lomb-Scargle periodogram
        # corresponds to amplitude of the generated input signal.
        dtype = xp.dtype(dtype)

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * xp.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        r = testing.shaped_random((nin,), xp, dtype=dtype, scale=1.0,
                                  seed=2353425)
        t = xp.linspace(0.01 * xp.pi, 10. * xp.pi, nin, dtype=dtype)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * xp.sin(w * t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = xp.linspace(0.01, 10., nout, dtype=dtype)

        # Calculate Lomb-Scargle periodogram
        pgram = scp.signal.lombscargle(t, x, f)

        # Normalize
        pgram = xp.sqrt(4 * pgram / t.shape[0])
        return pgram

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_precenter(self, dtype, xp, scp):
        # Test if precenter gives the same result as manually precentering.
        dtype = xp.dtype(dtype)

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * xp.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select
        offset = 0.15  # Offset to be subtracted in pre-centering

        # Randomly select a fraction of an array with timesteps
        r = testing.shaped_random((nin,), xp, dtype=dtype, scale=1.0,
                                  seed=2353425)
        t = xp.linspace(0.01 * xp.pi, 10. * xp.pi, nin, dtype=dtype)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * xp.sin(w * t + phi) + offset

        # Define the array of frequencies for which to compute the periodogram
        f = xp.linspace(0.01, 10., nout, dtype=dtype)

        # Calculate Lomb-Scargle periodogram
        pgram = scp.signal.lombscargle(t, x, f, precenter=True)
        pgram2 = scp.signal.lombscargle(t, x - x.mean(), f, precenter=False)

        # check if centering worked
        return pgram, pgram2

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_normalize(self, dtype, xp, scp):
        # Test normalize option of Lomb-Scarge.

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * xp.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        r = testing.shaped_random((nin,), xp, dtype=dtype, scale=1.0,
                                  seed=2353425)
        t = xp.linspace(0.01 * xp.pi, 10. * xp.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * xp.sin(w * t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = xp.linspace(0.01, 10., nout, dtype=dtype)

        # Calculate Lomb-Scargle periodogram
        pgram = scp.signal.lombscargle(t, x, f)
        pgram2 = scp.signal.lombscargle(t, x, f, normalize=True)

        # check if normalization works as expected
        return pgram, pgram2


@pytest.mark.xfail(
    runtime.is_hip and driver.get_build_version() < 5_00_00000,
    reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestPeriodogram:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_even(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_odd(self, xp, scp):
        x = xp.zeros(15)
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_twosided(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        f, p = scp.signal.periodogram(x, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_spectrum(self, xp, scp):
        x = np.zeros(16)
        x[0] = 1
        f, p = scp.signal.periodogram(x, scaling='spectrum')
        g, q = scp.signal.periodogram(x, scaling='density')
        return f, p, g, q

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_even(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_odd(self, xp, scp):
        x = xp.zeros(15, dtype=int)
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_twosided(self, xp, scp):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        f, p = scp.signal.periodogram(x, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_complex(self, xp, scp):
        x = xp.zeros(16, xp.complex128)
        x[0] = 1.0 + 2.0j
        f, p = scp.signal.periodogram(x, return_onesided=False)
        return f, p

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_unk_scaling(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            x = xp.zeros(4, xp.complex128)
            scp.signal.periodogram(x, scaling='foo')

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nd_axis_m1(self, xp, scp):
        x = xp.zeros(20, dtype=np.float64)
        x = x.reshape((2, 1, 10))
        x[:, :, 0] = 1.0
        f, p = scp.signal.periodogram(x)
        f0, p0 = scp.signal.periodogram(x[0, 0, :])
        return f, p, f0, p0

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nd_axis_0(self, xp, scp):
        x = xp.zeros(20, dtype=np.float64)
        x = x.reshape((10, 2, 1))
        x[0, :, :] = 1.0
        f, p = scp.signal.periodogram(x, axis=0)
        f0, p0 = scp.signal.periodogram(x[:, 0, 0])
        return f, p, f0, p0

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_window_external(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        f, p = scp.signal.periodogram(x, 10, 'hann')
        win = scp.signal.get_window('hann', 16)
        fe, pe = scp.signal.periodogram(x, 10, win)

        win_err = scp.signal.get_window('hann', 32)
        with pytest.raises(ValueError):
            scp.signal.periodogram(x, 10, win_err)

        return f, p, fe, pe

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_padded_fft(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        fp, pp = scp.signal.periodogram(x, nfft=32)
        return f, p, fp, pp

    @pytest.mark.parametrize('shape', [(0,), (3, 0), (0, 5, 2)])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_empty_input(self, shape, xp, scp):
        f, p = scp.signal.periodogram(xp.empty(shape))
        return f, p

    @pytest.mark.parametrize('shape', [(3, 0), (0, 5, 2)])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_empty_input_other_axis(self, shape, xp, scp):
        f, p = scp.signal.periodogram(xp.empty(shape), axis=1)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_short_nfft(self, xp, scp):
        x = xp.zeros(18)
        x[0] = 1
        f, p = scp.signal.periodogram(x, nfft=16)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nfft_is_xshape(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        f, p = scp.signal.periodogram(x, nfft=16)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_even_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_odd_32(self, xp, scp):
        x = xp.zeros(15, 'f')
        x[0] = 1
        f, p = scp.signal.periodogram(x)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_twosided_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        f, p = scp.signal.periodogram(x, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_complex_32(self, xp, scp):
        x = xp.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        f, p = scp.signal.periodogram(x, return_onesided=False)
        return f, p

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_shorter_window_error(self, mod):
        xp, scp = mod
        x = xp.zeros(16)
        x[0] = 1
        win = scp.signal.get_window('hann', 10)
        with pytest.raises(ValueError):
            scp.signal.periodogram(x, window=win)

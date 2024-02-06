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


@testing.with_requires('scipy')
class TestWelch:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_even(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_odd(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=9)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_twosided(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_spectrum(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8, scaling='spectrum')
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_onesided_even(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_onesided_odd(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=9)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_twosided(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_complex(self, xp, scp):
        x = xp.zeros(16, xp.complex128)
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = scp.signal.welch(x, nperseg=8, return_onesided=False)
        return f, p

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_unk_scaling(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            x = xp.zeros(4, xp.complex128)
            scp.signal.welch(x, scaling='foo', nperseg=4)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_linear(self, xp, scp):
        x = xp.arange(10, dtype=xp.float64) + 0.04
        f, p = scp.signal.welch(x, nperseg=10, detrend='linear')
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_no_detrending(self, xp, scp):
        x = xp.arange(10, dtype=xp.float64) + 0.04
        f1, p1 = scp.signal.welch(x, nperseg=10, detrend=False)
        f2, p2 = scp.signal.welch(x, nperseg=10, detrend=lambda x: x)
        return f1, p1, f2, p2

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_external(self, xp, scp):
        x = xp.arange(10, dtype=xp.float64) + 0.04
        f, p = scp.signal.welch(
            x, nperseg=10,
            detrend=lambda seg: scp.signal.detrend(seg, type='l'))
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_external_nd_m1(self, xp, scp):
        x = xp.arange(40, dtype=xp.float64) + 0.04
        x = x.reshape((2, 2, 10))
        f, p = scp.signal.welch(
            x, nperseg=10,
            detrend=lambda seg: scp.signal.detrend(seg, type='l'))
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_external_nd_0(self, xp, scp):
        x = xp.arange(20, dtype=xp.float64) + 0.04
        x = x.reshape((2, 1, 10))
        x = xp.moveaxis(x, 2, 0)
        f, p = scp.signal.welch(
            x, nperseg=10, axis=0,
            detrend=lambda seg: scp.signal.detrend(seg, axis=0, type='l'))
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nd_axis_m1(self, xp, scp):
        x = xp.arange(20, dtype=xp.float64) + 0.04
        x = x.reshape((2, 1, 10))
        f, p = scp.signal.welch(x, nperseg=10)
        f0, p0 = scp.signal.welch(x[0, 0, :], nperseg=10)
        return f, p, f0, p0

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nd_axis_0(self, xp, scp):
        x = xp.arange(20, dtype=xp.float64) + 0.04
        x = x.reshape((10, 2, 1))
        f, p = scp.signal.welch(x, nperseg=10, axis=0)
        f0, p0 = scp.signal.welch(x[:, 0, 0], nperseg=10)
        return f, p, f0, p0

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_window_external(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, 10, 'hann', nperseg=8)
        win = scp.signal.get_window('hann', 8)
        fe, pe = scp.signal.welch(x, 10, win, nperseg=None)

        with pytest.raises(ValueError):
            scp.signal.welch(x, 10, win, nperseg=4)

        with pytest.raises(ValueError):
            win_err = scp.signal.get_window('hann', 32)
            scp.signal.welch(x, 10, win_err, nperseg=None)

        return f, p, fe, pe

    @pytest.mark.parametrize('shape', [(0,), (3, 0), (0, 5, 2)])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_empty_input(self, shape, xp, scp):
        f, p = scp.signal.welch(xp.empty(shape))
        return f, p

    @pytest.mark.parametrize('shape', [(3, 0), (0, 5, 2)])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_empty_input_other_axis(self, shape, xp, scp):
        f, p = scp.signal.welch(xp.empty(shape), axis=1)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_short_data(self, xp, scp):
        x = np.zeros(8)
        x[0] = 1

        # default nperseg
        f, p = scp.signal.welch(x, window='hann')
        # user-specified nperseg
        f1, p1 = scp.signal.welch(x, window='hann', nperseg=256)
        # valid nperseg, doesn't give warning
        f2, p2 = scp.signal.welch(x, nperseg=8)
        return f, p, f1, p1, f2, p2

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_window_long_or_nd(self, mod):
        xp, scp = mod

        with pytest.raises(ValueError):
            scp.signal.welch(xp.zeros(4), 1, xp.array([1, 1, 1, 1, 1]))

        with pytest.raises(ValueError):
            scp.signal.welch(xp.zeros(4), 1, xp.arange(6).reshape((2, 3)))

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nondefault_noverlap(self, xp, scp):
        x = xp.zeros(64)
        x[::8] = 1
        f, p = scp.signal.welch(x, nperseg=16, noverlap=4)
        return f, p

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_bad_noverlap(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            scp.signal.welch(xp.zeros(4), 1, 'hann', 2, 7)

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_nfft_too_short(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            scp.signal.welch(xp.ones(12), nfft=3, nperseg=4)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_even_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_odd_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=9)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_twosided_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_complex_32(self, xp, scp):
        x = xp.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = scp.signal.welch(x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_padded_freqs(self, xp, scp):
        x = xp.zeros(12)

        nfft = 24
        fodd1, _ = scp.signal.welch(x, nperseg=5, nfft=nfft)
        feven1, _ = scp.signal.welch(x, nperseg=6, nfft=nfft)

        nfft = 25
        fodd2, _ = scp.signal.welch(x, nperseg=5, nfft=nfft)
        feven2, _ = scp.signal.welch(x, nperseg=6, nfft=nfft)

        return fodd1, feven1, fodd2, feven2

    @pytest.mark.parametrize(
        'window', ['hann', 'bartlett', ('tukey', 0.1), 'flattop'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_window_correction(self, window, xp, scp):
        A = 20
        fs = 1e4
        nperseg = int(fs // 10)
        fsig = 300

        tt = xp.arange(fs) / fs
        x = A * xp.sin(2 * np.pi * fsig * tt)

        _, p_spec = scp.signal.welch(
            x, fs=fs, nperseg=nperseg, window=window, scaling='spectrum')
        freq, p_dens = scp.signal.welch(
            x, fs=fs, nperseg=nperseg, window=window, scaling='density')

        return p_spec, freq, p_dens

    @pytest.mark.parametrize('axis', [0, 1, 2])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_axis_rolling(self, axis, xp, scp):
        x_flat = testing.shaped_random((1024,), xp)
        _, p_flat = scp.signal.welch(x_flat)

        newshape = [1,] * 3
        newshape[axis] = -1
        x = x_flat.reshape(newshape)

        _, p_plus = scp.signal.welch(x, axis=axis)  # Positive axis index
        # Negative axis index
        _, p_minus = scp.signal.welch(x, axis=axis - x.ndim)
        return p_flat, p_plus, p_minus

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_average(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.welch(x, nperseg=8, average='median')

        with pytest.raises(ValueError):
            scp.signal.welch(x, nperseg=8, average='unrecognised-average')
        return f, p


@testing.with_requires('scipy')
class TestCSD:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_pad_shorter_x(self, xp, scp):
        x = xp.zeros(8)
        y = xp.zeros(12)
        f1, c1 = scp.signal.csd(x, y, nperseg=12)
        return f1, c1

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_pad_shorter_y(self, xp, scp):
        x = xp.zeros(12)
        y = xp.zeros(8)
        f1, c1 = scp.signal.csd(x, y, nperseg=12)
        return f1, c1

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_even(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_odd(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=9)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_twosided(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_spectrum(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8, scaling='spectrum')
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_onesided_even(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_onesided_odd(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=9)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_integer_twosided(self, xp, scp):
        x = xp.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_complex(self, xp, scp):
        x = xp.zeros(16, xp.complex128)
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = scp.signal.csd(x, x, nperseg=8, return_onesided=False)
        return f, p

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_unk_scaling(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            scp.signal.csd(
                xp.zeros(4, np.complex128), xp.ones(4, np.complex128),
                scaling='foo', nperseg=4)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_linear(self, xp, scp):
        x = xp.arange(10, dtype=xp.float64) + 0.04
        f, p = scp.signal.csd(x, x, nperseg=10, detrend='linear')
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_no_detrending(self, xp, scp):
        x = xp.arange(10, dtype=xp.float64) + 0.04
        f1, p1 = scp.signal.csd(x, x, nperseg=10, detrend=False)
        f2, p2 = scp.signal.csd(x, x, nperseg=10, detrend=lambda x: x)
        return f1, p1, f2, p2

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_external(self, xp, scp):
        x = xp.arange(10, dtype=xp.float64) + 0.04
        f, p = scp.signal.csd(
            x, x, nperseg=10,
            detrend=lambda seg: scp.signal.detrend(seg, type='l'))
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_external_nd_m1(self, xp, scp):
        x = xp.arange(40, dtype=xp.float64) + 0.04
        x = x.reshape((2, 2, 10))
        f, p = scp.signal.csd(
            x, x, nperseg=10,
            detrend=lambda seg: scp.signal.detrend(seg, type='l'))
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_detrend_external_nd_0(self, xp, scp):
        x = xp.arange(20, dtype=xp.float64) + 0.04
        x = x.reshape((2, 1, 10))
        x = xp.moveaxis(x, 2, 0)
        f, p = scp.signal.csd(
            x, x, nperseg=10, axis=0,
            detrend=lambda seg: scp.signal.detrend(seg, axis=0, type='l'))
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nd_axis_m1(self, xp, scp):
        x = xp.arange(20, dtype=xp.float64) + 0.04
        x = x.reshape((2, 1, 10))
        f, p = scp.signal.csd(x, x, nperseg=10)
        f0, p0 = scp.signal.csd(x[0, 0, :], x[0, 0, :], nperseg=10)
        return f, p, f0, p0

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nd_axis_0(self, xp, scp):
        x = xp.arange(20, dtype=xp.float64) + 0.04
        x = x.reshape((10, 2, 1))
        f, p = scp.signal.csd(x, x, nperseg=10, axis=0)
        f0, p0 = scp.signal.csd(x[:, 0, 0], x[:, 0, 0], nperseg=10)
        return f, p, f0, p0

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_window_external(self, xp, scp):
        x = xp.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, 10, 'hann', 8)
        win = scp.signal.get_window('hann', 8)
        fe, pe = scp.signal.csd(x, x, 10, win, nperseg=None)
        win_err = scp.signal.get_window('hann', 32)

        with pytest.raises(ValueError):
            scp.signal.csd(x, x, 10, win_err, nperseg=None)
        return f, p, fe, pe

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_empty_input(self, xp, scp):
        result = []

        for shape in [(0,), (3, 0), (0, 5, 2)]:
            f, p = scp.signal.csd(xp.empty(shape), xp.empty(shape))
            result += [f, p]

        f, p = scp.signal.csd(xp.ones(10), xp.empty((5, 0)))
        result += [f, p]

        f, p = scp.signal.csd(xp.empty((5, 0)), xp.ones(10))
        result += [f, p]
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_empty_input_other_axis(self, xp, scp):
        result = []
        for shape in [(3, 0), (0, 5, 2)]:
            f, p = scp.signal.csd(xp.empty(shape), xp.empty(shape), axis=1)
            result += [f, p]

        f, p = scp.signal.csd(xp.empty((10, 10, 3)),
                              xp.zeros((10, 0, 1)), axis=1)
        result += [f, p]

        f, p = scp.signal.csd(xp.empty((10, 0, 1)),
                              xp.zeros((10, 10, 3)), axis=1)
        result += [f, p]
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_short_data(self, xp, scp):
        x = xp.zeros(8)
        x[0] = 1

        # default nperseg
        f, p = scp.signal.csd(x, x, window='hann')
        # user-specified nperseg
        f1, p1 = scp.signal.csd(x, x, window='hann', nperseg=256)
        # valid nperseg, doesn't give warning
        f2, p2 = scp.signal.csd(x, x, nperseg=8)
        return f, p, f1, p1, f2, p2

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_window_long_or_nd(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            scp.signal.csd(xp.zeros(4), xp.zeros(
                4), 1, xp.array([1, 1, 1, 1, 1]))

        with pytest.raises(ValueError):
            scp.signal.csd(xp.zeros(4), xp.ones(4), 1,
                           xp.arange(6).reshape((2, 3)))

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_nondefault_noverlap(self, xp, scp):
        x = xp.zeros(64)
        x[::8] = 1
        f, p = scp.signal.csd(x, x, nperseg=16, noverlap=4)
        return f, p

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_bad_noverlap(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            scp.signal.csd(xp.zeros(4), xp.ones(4), 1, 'hann', 2, 7)

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_nfft_too_short(self, mod):
        xp, scp = mod
        with pytest.raises(ValueError):
            scp.signal.csd(xp.ones(12), xp.zeros(12), nfft=3,
                           nperseg=4)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_even_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_onesided_odd_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=9)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_real_twosided_32(self, xp, scp):
        x = xp.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = scp.signal.csd(x, x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_complex_32(self, xp, scp):
        x = xp.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = scp.signal.csd(x, x, nperseg=8, return_onesided=False)
        return f, p

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_padded_freqs(self, xp, scp):
        x = xp.zeros(12)
        y = xp.ones(12)

        nfft = 24
        fodd1, _ = scp.signal.csd(x, y, nperseg=5, nfft=nfft)
        feven1, _ = scp.signal.csd(x, y, nperseg=6, nfft=nfft)

        nfft = 25
        fodd2, _ = scp.signal.csd(x, y, nperseg=5, nfft=nfft)
        feven2, _ = scp.signal.csd(x, y, nperseg=6, nfft=nfft)
        return fodd1, feven1, fodd2, feven2

    @pytest.mark.skipif(
        cupy.cuda.runtime.runtimeGetVersion() < 12000,
        reason="It fails on CUDA 11.x")
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_copied_data(self, xp, scp):
        x = testing.shaped_random((64,), xp, xp.float64)
        y = x.copy()

        _, p_same1 = scp.signal.csd(x, x, nperseg=8, average='mean',
                                    return_onesided=False)
        _, p_copied1 = scp.signal.csd(x, y, nperseg=8, average='mean',
                                      return_onesided=False)

        _, p_same2 = scp.signal.csd(x, x, nperseg=8, average='median',
                                    return_onesided=False)
        _, p_copied2 = scp.signal.csd(x, y, nperseg=8, average='median',
                                      return_onesided=False)
        return p_same1, p_copied1, p_same2, p_copied2


@testing.with_requires('scipy')
class TestCheckNOLACOLA:

    @pytest.mark.parametrize('setting', [
        ('boxcar', 10, 0),
        ('boxcar', 10, 9),
        ('bartlett', 51, 26),
        ('hann', 256, 128),
        ('hann', 256, 192),
        ('blackman', 300, 200),
        (('tukey', 0.5), 256, 64),
        ('hann', 256, 255),
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_check_COLA(self, setting, xp, scp):
        return scp.signal.check_COLA(*setting)

    @pytest.mark.parametrize('setting', [
        ('boxcar', 10, 0),
        ('boxcar', 10, 9),
        ('boxcar', 10, 7),
        ('bartlett', 51, 26),
        ('bartlett', 51, 10),
        ('hann', 256, 128),
        ('hann', 256, 192),
        ('hann', 256, 37),
        ('blackman', 300, 200),
        ('blackman', 300, 123),
        (('tukey', 0.5), 256, 64),
        (('tukey', 0.5), 256, 38),
        ('hann', 256, 255),
        ('hann', 256, 39),
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_check_NOLA(self, setting, xp, scp):
        return scp.signal.check_NOLA(*setting)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_check_NOLA_fail(self, xp, scp):
        w_fail = xp.ones(16)
        w_fail[::2] = 0

        settings_fail = [
            (w_fail, len(w_fail), len(w_fail) // 2),
            ('hann', 64, 0),
        ]
        result = []
        for setting in settings_fail:
            result.append(scp.signal.check_NOLA(*setting))
        return result


@testing.with_requires('scipy>=1.9.0')
class TestSTFT:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_average_all_segments(self, xp, scp):
        x = testing.shaped_random((1024,), xp, xp.float64, scale=1, seed=1234)

        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8

        # Compare twosided, because onesided welch doubles non-DC terms to
        # account for power at negative frequencies. stft doesn't do this,
        # because it breaks invertibility.
        f, _, Z = scp.signal.stft(
            x, fs, window, nperseg, noverlap, padded=False,
            return_onesided=False, boundary=None)

        return f, Z

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_permute_axes(self, xp, scp):
        x = testing.shaped_random((1024,), xp, xp.float64, scale=1, seed=1234)

        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8

        f1, t1, Z1 = scp.signal.stft(x, fs, window, nperseg, noverlap)
        f2, t2, Z2 = scp.signal.stft(
            x.reshape((-1, 1, 1)), fs, window, nperseg, noverlap, axis=0)

        t3, x1 = scp.signal.istft(Z1, fs, window, nperseg, noverlap)
        t4, x2 = scp.signal.istft(
            Z2.T, fs, window, nperseg, noverlap, time_axis=0, freq_axis=-1)

        return f1, t1, Z1, f2, t2, Z2, t3, x1, t4, x2.reshape(-1)

    @pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
    @pytest.mark.parametrize('settings', [
        ('boxcar', 100, 10, 0),           # Test no overlap
        ('boxcar', 100, 10, 9),           # Test high overlap
        ('bartlett', 101, 51, 26),        # Test odd nperseg
        ('hann', 1024, 256, 128),         # Test defaults
        (('tukey', 0.5), 1152, 256, 64),  # Test Tukey
        ('hann', 1024, 256, 255),         # Test overlapped hann
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roundtrip_real(self, scaling, settings, xp, scp):
        window, N, nperseg, noverlap = settings
        t = xp.arange(N)
        x = testing.shaped_random(t.shape, xp, xp.float64, seed=1234)

        _, _, zz = scp.signal.stft(
            x, nperseg=nperseg, noverlap=noverlap,
            window=window, detrend=None, padded=False,
            scaling=scaling)

        tr, xr = scp.signal.istft(
            zz, nperseg=nperseg, noverlap=noverlap, window=window,
            scaling=scaling)

        return zz, tr, xr

    @pytest.mark.parametrize('settings', [
        ('hann', 1024, 256, 128)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_roundtrip_float32(self, settings, xp, scp):
        window, N, nperseg, noverlap = settings
        t = xp.arange(N)
        x = testing.shaped_random(t.shape, xp, xp.float32, seed=1234)

        _, _, zz = scp.signal.stft(
            x, nperseg=nperseg, noverlap=noverlap,
            window=window, detrend=None, padded=False)

        tr, xr = scp.signal.istft(zz, nperseg=nperseg, noverlap=noverlap,
                                  window=window)

        return zz, tr, xr

    @pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
    @pytest.mark.parametrize('settings', [
        ('boxcar', 100, 10, 0),           # Test no overlap
        ('boxcar', 100, 10, 9),           # Test high overlap
        ('bartlett', 101, 51, 26),        # Test odd nperseg
        ('hann', 1024, 256, 128),         # Test defaults
        (('tukey', 0.5), 1152, 256, 64),  # Test Tukey
        ('hann', 1024, 256, 255),         # Test overlapped hann
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roundtrip_complex(self, scaling, settings, xp, scp):
        window, N, nperseg, noverlap = settings
        t = np.arange(N)
        x_real = testing.shaped_random(t.shape, xp, xp.float64, seed=1234)
        x_imag = 1j * testing.shaped_random(t.shape, xp, xp.float64)
        x = x_real + x_imag

        _, _, zz = scp.signal.stft(
            x, nperseg=nperseg, noverlap=noverlap,
            window=window, detrend=None, padded=False,
            return_onesided=False, scaling=scaling)

        tr, xr = scp.signal.istft(
            zz, nperseg=nperseg, noverlap=noverlap,
            window=window, input_onesided=False, scaling=scaling)

        return zz, tr, xr

    @pytest.mark.parametrize('settings', [
        ('boxcar', 100, 10, 0),           # Test no overlap
        ('boxcar', 100, 10, 9),           # Test high overlap
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roundtrip_boundary_extension(self, settings, xp, scp):
        # Test against boxcar, since window is all ones, and thus can be fully
        # recovered with no boundary extension
        window, N, nperseg, noverlap = settings
        t = xp.arange(N)
        x = testing.shaped_random(t.shape, xp, xp.float64, seed=1234)

        results = []
        _, _, zz = scp.signal.stft(
            x, nperseg=nperseg, noverlap=noverlap,
            window=window, detrend=None, padded=True, boundary=None)

        _, xr = scp.signal.istft(
            zz, noverlap=noverlap, window=window, boundary=False)

        results.append(zz)
        results.append(xr)

        for boundary in ['even', 'odd', 'constant', 'zeros']:
            _, _, zz_ext = scp.signal.stft(
                x, nperseg=nperseg, noverlap=noverlap,
                window=window, detrend=None, padded=True,
                boundary=boundary)

            _, xr_ext = scp.signal.istft(
                zz_ext, noverlap=noverlap, window=window, boundary=True)

            results.append(zz_ext)
            results.append(xr_ext)

        return results

    @pytest.mark.parametrize('settings', [
        ('boxcar', 101, 10, 0),
        ('hann', 1000, 256, 128),
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7)
    def test_roundtrip_padded_signal(self, settings, xp, scp):
        window, N, nperseg, noverlap = settings
        t = xp.arange(N)
        x = testing.shaped_random(t.shape, xp, xp.float64, seed=1234)

        _, _, zz = scp.signal.stft(
            x, nperseg=nperseg, noverlap=noverlap,
            window=window, detrend=None, padded=True)

        tr, xr = scp.signal.istft(zz, noverlap=noverlap, window=window)
        return zz, tr, xr

    @pytest.mark.parametrize('settings', [
        ('hann', 1024, 256, 128, 512),
        ('hann', 1024, 256, 128, 501),
        ('boxcar', 100, 10, 0, 33),
        (('tukey', 0.5), 1152, 256, 64, 1024),
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roundtrip_padded_FFT(self, settings, xp, scp):
        window, N, nperseg, noverlap, nfft = settings
        t = np.arange(N)
        x = testing.shaped_random(t.shape, xp, xp.float64, seed=1234)
        xc = x * xp.exp(1j * xp.pi / 4)

        # real signal
        _, _, z = scp.signal.stft(
            x, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window=window, detrend=None, padded=True)

        # complex signal
        _, _, zc = scp.signal.stft(
            xc, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window=window, detrend=None, padded=True, return_onesided=False)

        tr, xr = scp.signal.istft(
            z, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window)

        tcr, xcr = scp.signal.istft(
            zc, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window=window, input_onesided=False)
        return z, zc, xr, xcr, tr, tcr

    @pytest.mark.parametrize('a', [0, 1, 2])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_axis_rolling(self, a, xp, scp):
        x_flat = testing.shaped_random((1024,), xp, xp.float64, seed=1234)
        _, _, z_flat = scp.signal.stft(x_flat)

        newshape = [1,]*3
        newshape[a] = -1
        x = x_flat.reshape(newshape)

        # Positive axis index
        _, _, z_plus = scp.signal.stft(x, axis=a)
        # Negative axis index
        _, _, z_minus = scp.signal.stft(x, axis=a - x.ndim)

        # z_flat has shape [n_freq, n_time]

        # Test vs. transpose
        _, x_transpose_m = scp.signal.istft(
            z_flat.T, time_axis=-2, freq_axis=-1)
        _, x_transpose_p = scp.signal.istft(
            z_flat.T, time_axis=0, freq_axis=1)
        return z_plus, z_minus, x_transpose_m, x_transpose_p

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-7)
    def test_roundtrip_scaling(self, xp, scp):
        """Verify behavior of scaling parameter. """
        # Create 1024 sample cosine signal with amplitude 2:
        X = xp.zeros(513, dtype=complex)
        X[256] = 1024
        x = xp.fft.irfft(X)

        results = []

        # Calculate magnitude-scaled STFT:
        Zs = scp.signal.stft(x, boundary='even', scaling='spectrum')[2]
        results.append(Zs)

        # Test round trip:
        x1 = scp.signal.istft(Zs, boundary=True, scaling='spectrum')[1]
        results.append(x1)

        # Calculate two-sided psd-scaled STFT:
        #  - using 'even' padding since signal is axis symmetric - this ensures
        #    stationary behavior on the boundaries
        #  - using the two-sided transform allows determining the spectral
        #    power by `sum(abs(Zp[:, k])**2) / len(f)` for the k-th time slot.
        Zp = scp.signal.stft(
            x, return_onesided=False, boundary='even', scaling='psd')[2]
        results.append(Zp)

        # Test round trip:
        x1 = scp.signal.istft(
            Zp, input_onesided=False, boundary=True, scaling='psd')[1]
        results.append(x1)

        # The power of the one-sided psd-scaled STFT can be determined
        # analogously (note that the two sides are not of equal shape):
        Zp0 = scp.signal.stft(
            x, return_onesided=True, boundary='even', scaling='psd')[2]
        results.append(Zp0)

        # Test round trip:
        x1 = scp.signal.istft(
            Zp0, input_onesided=True, boundary=True, scaling='psd')[1]
        results.append(x1)
        return results


@testing.with_requires('scipy')
class TestVectorstrength:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_single_1dperiod(self, xp, scp):
        events = xp.array([.5])
        period = 5.

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_single_2dperiod(self, xp, scp):
        events = xp.array([.5])
        period = [1, 2, 5.]

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_equal_1dperiod(self, xp, scp):
        events = xp.array([.25, .25, .25, .25, .25, .25])
        period = 2
        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_equal_2dperiod(self, xp, scp):
        events = xp.array([.25, .25, .25, .25, .25, .25])
        period = [1, 2, ]

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_spaced_1dperiod(self, xp, scp):
        events = xp.array([.1, 1.1, 2.1, 4.1, 10.1])
        period = 1

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_spaced_2dperiod(self, xp, scp):
        events = xp.array([.1, 1.1, 2.1, 4.1, 10.1])
        period = [1, .5]

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_partial_1dperiod(self, xp, scp):
        events = xp.array([.25, .5, .75])
        period = 1

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_partial_2dperiod(self, xp, scp):
        events = xp.array([.25, .5, .75])
        period = [1., 1., 1., 1.]

        strength, phase = scp.signal.vectorstrength(events, period)
        return strength, phase

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-7, atol=1e-7)
    def test_opposite_1dperiod(self, xp, scp):
        events = xp.array([0, .25, .5, .75])
        period = 1.
        strength, _ = scp.signal.vectorstrength(events, period)
        return strength

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-7, atol=1e-7)
    def test_opposite_2dperiod(self, xp, scp):
        events = xp.array([0, .25, .5, .75])
        period = [1.] * 10
        strength, _ = scp.signal.vectorstrength(events, period)
        return strength

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_2d_events_ValueError(self, mod):
        xp, scp = mod
        events = xp.array([[1, 2]])
        period = 1.
        with pytest.raises(ValueError):
            scp.signal.vectorstrength(events, period)

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_2d_period_ValueError(self, mod):
        xp, scp = mod
        events = 1.
        period = xp.array([[1]])
        with pytest.raises(ValueError):
            scp.signal.vectorstrength(events, period)

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_zero_period_ValueError(self, mod):
        _, scp = mod
        events = 1.
        period = 0
        with pytest.raises(ValueError):
            scp.signal.vectorstrength(events, period)

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_negative_period_ValueError(self, mod):
        _, scp = mod
        events = 1.
        period = -1
        with pytest.raises(ValueError):
            scp.signal.vectorstrength(events, period)


@testing.with_requires('scipy')
class TestCoherence:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_identical_input(self, xp, scp):
        x = testing.shaped_random((20,), xp, xp.float64, scale=1.0)
        y = xp.copy(x)  # So `y is x` -> False

        f1, C1 = scp.signal.coherence(x, y, nperseg=10)
        return f1, C1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_phase_shifted_input(self, xp, scp):
        x = testing.shaped_random((20,), xp, xp.float64, scale=1.0)
        y = -x

        f1, C1 = scp.signal.coherence(x, y, nperseg=10)
        return f1, C1


@testing.with_requires('scipy')
class TestSpectrogram:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_average_all_segments(self, xp, scp):
        x = testing.shaped_random((1024,), xp, xp.float64, scale=1.0)

        fs = 1.0
        window = ('tukey', 0.25)
        nperseg = 16
        noverlap = 2

        f, _, P = scp.signal.spectrogram(x, fs, window, nperseg, noverlap)
        return f, P

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_window_external(self, xp, scp):
        x = testing.shaped_random((1024,), xp, xp.float64, scale=1.0)

        fs = 1.0
        win = scp.signal.get_window(('tukey', 0.25), 16)
        fe, _, Pe = scp.signal.spectrogram(
            x, fs, win, nperseg=None, noverlap=2)

        with pytest.raises(ValueError):
            scp.signal.spectrogram(x, fs, win, nperseg=8)

        win_err = scp.signal.get_window(('tukey', 0.25), 2048)
        with pytest.raises(ValueError):
            scp.signal.spectrogram(x, fs, win_err, nperseg=None)
        return fe, Pe

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_short_data(self, xp, scp):
        x = testing.shaped_random((1024,), xp, xp.float64, scale=1.0)
        fs = 1.0

        # for string-like window, input signal length < nperseg value gives
        # UserWarning, sets nperseg to x.shape[-1]
        # default nperseg
        f, _, p = scp.signal.spectrogram(x, fs, window=('tukey', 0.25))

        # user-specified nperseg
        f1, _, p1 = scp.signal.spectrogram(
            x, fs, window=('tukey', 0.25), nperseg=1025)
        # to compare w/default
        f2, _, p2 = scp.signal.spectrogram(x, fs, nperseg=256)
        # compare w/user-spec'd
        f3, _, p3 = scp.signal.spectrogram(x, fs, nperseg=1024)
        return f, p, f1, p1, f2, p2, f3, p3

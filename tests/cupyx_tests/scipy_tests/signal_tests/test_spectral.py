
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

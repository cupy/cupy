
import pytest

import numpy as np

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

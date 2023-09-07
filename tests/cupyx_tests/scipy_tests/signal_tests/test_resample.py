
import cupy
from cupy import testing

import cupyx.scipy.signal  # NOQA
import cupyx.scipy.signal.windows  # NOQA

try:
    import scipy.signal  # NOQA
    import scipy.signal.windows  # NOQA
except ImportError:
    pass

import pytest
from math import gcd
from itertools import product

padtype_options = ["mean", "median", "minimum", "maximum", "line"]

_upfirdn_modes = [
    'constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect',
    'antisymmetric', 'antireflect', 'line',
]

padtype_options += _upfirdn_modes


@testing.with_requires('scipy')
class TestResample:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        # Some basic tests

        # Regression test for issue #scipy/3603.
        # window.shape must equal to sig.shape[0]
        sig = xp.arange(128)
        num = 256
        win = scp.signal.get_window(('kaiser', 8.0), 160)

        with pytest.raises(ValueError):
            scp.signal.resample(sig, num, window=win)

        # Other degenerate conditions
        with pytest.raises(ValueError):
            scp.signal.resample_poly(sig, 'yo', 1)

        with pytest.raises(ValueError):
            scp.signal.resample_poly(sig, 1, 0)

        with pytest.raises(ValueError):
            scp.signal.resample_poly(sig, 2, 1, padtype='')

        with pytest.raises(ValueError):
            scp.signal.resample_poly(sig, 2, 1, padtype='mean', cval=10)

        # test for issue #scipy/6505 - should not modify window.shape
        # when axis ≠ 0
        sig2 = xp.tile(xp.arange(160), (2, 1))
        return scp.signal.resample(sig2, num, axis=-1, window=win).copy()

    @pytest.mark.parametrize('window', (None, 'hamming'))
    @pytest.mark.parametrize('N', (20, 19))
    @pytest.mark.parametrize('num', (100, 101, 10, 11))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rfft(self, N, num, window, xp, scp):
        # Make sure the speed up using rfft gives the same result as the normal
        # way using fft
        results = []
        x = xp.linspace(0, 10, N, endpoint=False)
        y = xp.cos(-x ** 2 / 6.0)

        results.append(scp.signal.resample(y, num, window=window))
        results.append(scp.signal.resample(y + 0j, num, window=window).real)

        y = xp.array([xp.cos(-x**2 / 6.0), xp.sin(-x**2 / 6.0)])
        y_complex = y + 0j
        results.append(scp.signal.resample(y, num, axis=1, window=window))
        results.append(
            scp.signal.resample(y_complex, num, axis=1, window=window).real)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_input_domain(self, xp, scp):
        # Test if both input domain modes produce the same results.
        tsig = xp.arange(256) + 0j
        fsig = xp.fft.fft(tsig)
        num = 256
        return (scp.signal.resample(fsig, num, domain='freq'),
                scp.signal.resample(tsig, num, domain='time'))

    @pytest.mark.parametrize('nx', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('ny', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('dtype', ('float', 'complex'))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dc(self, nx, ny, dtype, xp, scp):
        x = xp.array([1] * nx, dtype)
        y = scp.signal.resample(x, ny)
        return y

    @pytest.mark.parametrize('padtype', padtype_options)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_mutable_window(self, padtype, xp, scp):
        # Test that a mutable window is not modified
        impulse = xp.zeros(3)
        window = xp.random.RandomState(0).randn(2)
        scp.signal.resample_poly(impulse, 5, 1, window=window, padtype=padtype)
        # assert_array_equal(window, window_orig)
        return window

    @pytest.mark.parametrize('padtype', padtype_options)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_output_float32(self, padtype, xp, scp):
        # Test that float32 inputs yield a float32 output
        x = xp.arange(10, dtype=xp.float32)
        h = xp.array([1, 1, 1], dtype=xp.float32)
        y = scp.signal.resample_poly(x, 1, 2, window=h, padtype=padtype)
        return y

    @pytest.mark.parametrize('padtype', padtype_options)
    @pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_output_match_dtype(self, padtype, dtype, xp, scp):
        # Test that the dtype of x is preserved per issue #14733
        x = xp.arange(10, dtype=dtype)
        y = scp.signal.resample_poly(x, 1, 2, padtype=padtype)
        return y

    @pytest.mark.parametrize(
        "method, ext, padtype",
        [("fft", False, None)]
        + list(
            product(
                ["polyphase"], [False, True], padtype_options,
            )
        ),
    )
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_resample_methods(self, method, ext, padtype, xp, scp):
        # Test resampling of sinusoids and random noise (1-sec)
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]

        # Sinusoids, windowed to avoid edge artifacts
        t = xp.arange(rate) / float(rate)
        freqs = xp.array((1., 10., 40.))[:, xp.newaxis]
        x = xp.sin(2 * xp.pi * freqs * t) * scp.signal.windows.hann(rate)
        results = []

        for rate_to in rates_to:
            if method == 'fft':
                y_resamps = scp.signal.resample(x, rate_to, axis=-1)
            else:
                if ext and rate_to != rate:
                    # Match default window design
                    g = gcd(rate_to, rate)
                    up = rate_to // g
                    down = rate // g
                    max_rate = max(up, down)
                    f_c = 1. / max_rate
                    half_len = 10 * max_rate
                    window = scp.signal.firwin(
                        2 * half_len + 1, f_c, window=('kaiser', 5.0))
                    polyargs = {'window': window, 'padtype': padtype}
                else:
                    polyargs = {'padtype': padtype}

                y_resamps = scp.signal.resample_poly(
                    x, rate_to, rate, axis=-1, **polyargs)

            results.append(y_resamps)

        # Random data
        rng = xp.random.RandomState(0)
        # low-pass, wind
        x = scp.signal.windows.hann(rate) * xp.cumsum(rng.randn(rate))
        for rate_to in rates_to:
            # random data
            if method == 'fft':
                y_resamp = scp.signal.resample(x, rate_to)
            else:
                y_resamp = scp.signal.resample_poly(
                    x, rate_to, rate, padtype=padtype)
            results.append(y_resamp)

        # More tests of fft method (Master 0.18.1 fails these)
        if method == 'fft':
            x1 = xp.array([1.+0.j, 0.+0.j])
            y1_test = scp.signal.resample(x1, 4)
            x2 = xp.array([1., 0.5, 0., 0.5])
            y2_test = scp.signal.resample(x2, 2)  # downsampling a real array
            results.append(y1_test)
            results.append(y2_test)

        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_poly_vs_filtfilt(self, xp, scp):
        # Check that up=1.0 gives same answer as filtfilt + slicing
        random_state = xp.random.RandomState(17)
        try_types = (int, xp.float32, xp.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]
        results = []

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (xp.complex64, xp.complex128):
                x += 1j * random_state.randn(size)

            # resample_poly assumes zeros outside of signl, whereas filtfilt
            # can only constant-pad. Make them equivalent:
            x[0] = 0
            x[-1] = 0

            for down in down_factors:
                h = scp.signal.firwin(31, 1. / down, window='hamming')

                # Need to pass convolved version of filter to resample_poly,
                # since filtfilt does forward and backward, but resample_poly
                # only goes forward
                hc = scp.signal.convolve(h, h[::-1])
                y = scp.signal.resample_poly(x, 1, down, window=hc)
                results.append(y)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_correlate1d(self, xp, scp):
        results = []
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = xp.random.random((nx,))
                    weights = xp.random.random((nweights,))
                    y_s = scp.signal.resample_poly(
                        x, up=1, down=down, window=weights)
                    results.append(y_s)
        return results

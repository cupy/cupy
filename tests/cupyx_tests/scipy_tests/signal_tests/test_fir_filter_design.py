import platform

import cupy

import cupyx.scipy.signal as signal
from cupy import testing

import pytest
from pytest import raises as assert_raises


@testing.with_requires("scipy")
class TestKaiser:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kaiser_beta(self, xp, scp):
        k = scp.signal.kaiser_beta
        return k(58.7), k(22.0), k(21.0), k(10.0)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kaiser_atten(self, xp, scp):
        k = scp.signal.kaiser_atten
        return k(1, 1.0), k(2, 1.0 / xp.pi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kaiserord(self, xp, scp):
        assert_raises(ValueError, scp.signal.kaiserord, 1.0, 1.0)

        return scp.signal.kaiserord(2.285 + 7.95 - 0.001, 1 / xp.pi)


@testing.with_requires('scipy')
class TestFirwin:
    @pytest.mark.parametrize('args, kwds', [
        ((51, .5), dict()),   # low-pass from 0 to f
        ((52, .5), dict(window='nuttall')),  # specific window
        ((53, .5), dict(pass_zero=False)),  # stop from 0 to f --> high-pass
        ((54, [.2, .4]), dict(pass_zero=False)),  # band-pass filter
        ((55, [.2, .4]), dict()),  # band-stop filter
        ((56, [.2, .4, .6, .8]), dict(pass_zero=False, scale=False)),
        ((57, [.2, .4, .6, .8]), dict()),   # multiband filter
        ((58, 0.1), dict(width=.03)),  # low-pass
        ((59, 0.1), dict(pass_zero=False)),  # high-pass
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_response(self, xp, scp, args, kwds):
        h = scp.signal.firwin(*args, **kwds)
        return h

    @pytest.mark.parametrize('case', [
        ([.5], True, (0, 1)),
        ([0.2, .6], False, (.4, 1)),
        ([.5], False, (1, 1)),
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_scaling(self, xp, scp, case):
        """
        For one lowpass, bandpass, and highpass example filter, this test
        checks two things:
          - the mean squared error over the frequency domain of the unscaled
            filter is smaller than the scaled filter (true for rectangular
            window)
          - the response of the scaled filter is exactly unity at the center
            of the first passband
        """
        N = 11
        cutoff, pass_zero, expected_responce = case
        fw = scp.signal.firwin
        h = fw(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
        hs = fw(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
        return h, hs

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lowpass(self, xp, scp):
        width = 0.04
        ntaps, beta = scp.signal.kaiserord(120, width)
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, **kwargs)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp):
        width = 0.04
        ntaps, beta = scp.signal.kaiserord(120, width)

        # Ensure that ntaps is odd.
        ntaps |= 1

        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, pass_zero=False, **kwargs)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        width = 0.04
        ntaps, beta = scp.signal.kaiserord(120, width)
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, pass_zero=False, **kwargs)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop_multi(self, xp, scp):
        width = 0.04
        ntaps, beta = scp.signal.kaiserord(120, width)
        kwargs = dict(cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta),
                      scale=False)
        taps = scp.signal.firwin(ntaps, **kwargs)
        return taps

    def test_bad_cutoff(self):
        """Test that invalid cutoff argument raises ValueError."""
        # cutoff values must be greater than 0 and less than 1.
        assert_raises(ValueError, signal.firwin, 99, -0.5)
        assert_raises(ValueError, signal.firwin, 99, 1.5)
        # Don't allow 0 or 1 in cutoff.
        assert_raises(ValueError, signal.firwin, 99, [0, 0.5])
        assert_raises(ValueError, signal.firwin, 99, [0.5, 1])
        # cutoff values must be strictly increasing.
        assert_raises(ValueError, signal.firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, signal.firwin, 99, [0.1, 0.5, 0.5])
        # Must have at least one cutoff value.
        assert_raises(ValueError, signal.firwin, 99, [])
        # 2D array not allowed.
        assert_raises(ValueError, signal.firwin, 99, [[0.1, 0.2], [0.3, 0.4]])
        # cutoff values must be less than nyq.
        assert_raises(ValueError, signal.firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, signal.firwin, 99, [10, 20, 30], fs=50)

    def test_even_highpass_raises_value_error(self):
        """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
        assert_raises(ValueError, signal.firwin, 40, 0.5, pass_zero=False)
        assert_raises(ValueError, signal.firwin, 40, [.25, 0.5])

    def test_bad_pass_zero(self):
        """Test degenerate pass_zero cases."""
        with assert_raises(ValueError, match='pass_zero must be'):
            signal.firwin(41, 0.5, pass_zero='foo')
        with assert_raises(TypeError):
            signal.firwin(41, 0.5, pass_zero=1.)
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                signal.firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                signal.firwin(41, [0.5], pass_zero=pass_zero)


@testing.with_requires('scipy')
class TestFirwin2:

    def test_invalid_args(self):
        # `freq` and `gain` have different lengths.
        with assert_raises(ValueError, match='must be of same length'):
            signal.firwin2(50, [0, 0.5, 1], [0.0, 1.0])
        # `nfreqs` is less than `ntaps`.
        with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
            signal.firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
        # Decreasing value in `freq`
        with assert_raises(ValueError, match='must be nondecreasing'):
            signal.firwin2(50, [0, 0.5, 0.4, 1.0], [0, .25, .5, 1.0])
        # Value in `freq` repeated more than once.
        with assert_raises(ValueError, match='must not occur more than twice'):
            signal.firwin2(50, [0, .1, .1, .1, 1.0], [
                           0.0, 0.5, 0.75, 1.0, 1.0])
        # `freq` does not start at 0.0.
        with assert_raises(ValueError, match='start with 0'):
            signal.firwin2(50, [0.5, 1.0], [0.0, 1.0])
        # `freq` does not end at fs/2.
        with assert_raises(ValueError, match='end with fs/2'):
            signal.firwin2(50, [0.0, 0.5], [0.0, 1.0])
        # Value 0 is repeated in `freq`
        with assert_raises(ValueError, match='0 must not be repeated'):
            signal.firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        # Value fs/2 is repeated in `freq`
        with assert_raises(ValueError, match='fs/2 must not be repeated'):
            signal.firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
        # Value in `freq` that is too close to a repeated number
        with assert_raises(ValueError, match='cannot contain numbers '
                                             'that are too close'):
            eps = cupy.finfo(float).eps
            signal.firwin2(50, [0.0, 0.5 - eps * 0.5, 0.5, 0.5, 1.0],
                           [1.0, 1.0, 1.0, 0.0, 0.0])

        # Type II filter, but the gain at nyquist frequency is not zero.
        with assert_raises(ValueError, match='Type II filter'):
            signal.firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])

        # Type III filter, but the gains at nyquist and zero rate are not zero.
        with assert_raises(ValueError, match='Type III filter'):
            signal.firwin2(17, [0.0, 0.5, 1.0], [
                           0.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            signal.firwin2(17, [0.0, 0.5, 1.0], [
                           1.0, 1.0, 0.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            signal.firwin2(17, [0.0, 0.5, 1.0], [
                           1.0, 1.0, 1.0], antisymmetric=True)

        # Type IV filter, but the gain at zero rate is not zero.
        with assert_raises(ValueError, match='Type IV filter'):
            signal.firwin2(16, [0.0, 0.5, 1.0], [
                           1.0, 1.0, 0.0], antisymmetric=True)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test01(self, xp, scp):
        beta = 12.0
        ntaps = 400
        # Filter is 1 from w=0 to w=0.5, then decreases linearly from 1 to 0
        # as w increases from w=0.5 to w=1  (w=1 is the Nyquist frequency).
        freq = xp.asarray([0.0, 0.5, 1.0])
        gain = xp.asarray([1.0, 1.0, 0.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=('kaiser', beta))
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test02(self, xp, scp):
        beta = 12.0
        # ntaps must be odd for positive gain at Nyquist.
        ntaps = 401
        # An ideal highpass filter.
        freq = xp.asarray([0.0, 0.5, 0.5, 1.0])
        gain = xp.asarray([0.0, 0.0, 1.0, 1.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=('kaiser', beta))
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test03(self, xp, scp):
        width = 0.02
        ntaps, beta = scp.signal.kaiserord(120, width)
        # ntaps must be odd for positive gain at Nyquist.
        ntaps = int(ntaps) | 1
        freq = xp.asarray([0.0, 0.4, 0.4, 0.5, 0.5, 1.0])
        gain = xp.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=('kaiser', beta))
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test04(self, xp, scp):
        """Test firwin2 when window=None."""
        ntaps = 5
        # Ideal lowpass: gain is 1 on [0,0.5], and 0 on [0.5, 1.0]
        freq = xp.asarray([0.0, 0.5, 0.5, 1.0])
        gain = xp.asarray([1.0, 1.0, 0.0, 0.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test05(self, xp, scp):
        """Test firwin2 for calculating Type IV filters"""
        ntaps = 1500

        freq = xp.asarray([0.0, 1.0])
        gain = xp.asarray([0.0, 1.0])
        taps = scp.signal.firwin2(
            ntaps, freq, gain, window=None, antisymmetric=True)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test06(self, xp, scp):
        """Test firwin2 for calculating Type III filters"""
        ntaps = 1501

        freq = xp.asarray([0.0, 0.5, 0.55, 1.0])
        gain = xp.asarray([0.0, 0.5, 0.0, 0.0])
        taps = scp.signal.firwin2(
            ntaps, freq, gain, window=None, antisymmetric=True)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_nyq(self, xp, scp):
        taps1 = scp.signal.firwin2(80,
                                   xp.asarray([0.0, 0.5, 1.0]),
                                   xp.asarray([1.0, 1.0, 0.0]))
        taps2 = scp.signal.firwin2(80,
                                   xp.asarray([0.0, 30.0, 60.0]),
                                   xp.asarray([1.0, 1.0, 0.0]), fs=120.0)
        return taps1, taps2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tuple(self, xp, scp):
        taps1 = scp.signal.firwin2(150,
                                   xp.asarray((0.0, 0.5, 0.5, 1.0)),
                                   xp.asarray((1.0, 1.0, 0.0, 0.0)))
        taps2 = scp.signal.firwin2(150,
                                   xp.asarray([0.0, 0.5, 0.5, 1.0]),
                                   xp.asarray([1.0, 1.0, 0.0, 0.0]))
        return taps1, taps2

    def test_input_modyfication(self):
        freq1 = cupy.array([0.0, 0.5, 0.5, 1.0])
        freq2 = cupy.array(freq1, copy=True)
        signal.firwin2(80, freq1, cupy.array([1.0, 1.0, 0.0, 0.0]))
        assert (freq1 == freq2).all()


@testing.with_requires('scipy')
class TestFirls:

    def test_bad_args(self):
        firls = signal.firls

        # even numtaps
        assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
        # odd bands
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
        # len(bands) != len(desired)
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
        # non-monotonic bands
        # assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
        # assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
        # assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
        # assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
        # negative desired
        # assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
        # len(weight) != len(pairs)
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [1, 2])
        # negative weight
        # assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [-1])

    @pytest.mark.xfail(
        platform.processor() == "aarch64",
        reason="aarch64 scipy does not match cupy/x86 see Scipy #20160")
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-13)
    def test_firls(self, xp, scp):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band
        # design a halfband symmetric low-pass filter
        h = scp.signal.firls(N, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)
        return h

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_firls_freqz(self, xp, scp):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band

        # design a halfband symmetric low-pass filter and check
        # the freq response
        h = scp.signal.firls(N, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)
        w, H = scp.signal.freqz(h, 1)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare(self, xp, scp):
        # compare to OCTAVE output
        taps = scp.signal.firls(
            9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], weight=[1, 2])
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_2(self, xp, scp):
        # compare to MATLAB output
        taps = scp.signal.firls(
            11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], weight=[1, 2])
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_3(self, xp, scp):
        # With linear changes:
        taps = scp.signal.firls(7, (0, 1, 2, 3, 4, 5), [
                                1, 0, 0, 1, 1, 0], fs=20)
        return taps

    @pytest.mark.xfail(reason="https://github.com/scipy/scipy/issues/18533")
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rank_deficient(self, xp, scp):
        # solve() runs but warns (only sometimes, so here we don't use match)
        x = scp.signal.firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        w, h = scp.signal.freqz(x, fs=2.)
        return x, w, h

    @pytest.mark.xfail(reason="https://github.com/scipy/scipy/issues/18533")
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rank_deficient_2(self, xp, scp):
        # switch to pinvh (tolerances could be higher with longer
        # filters, but using shorter ones is faster computationally and
        # the idea is the same)
        x = scp.signal.firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = scp.signal.freqz(x, fs=2.)
        return x, w, h

    def test_rank_deficient_3(self):
        # the same test as in scipy.signal
        x = signal.firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = signal.freqz(x, fs=2.)

        mask = w < 0.01
        assert mask.sum() > 3
        testing.assert_allclose(cupy.abs(h[mask]), 1., atol=1e-4)

        mask = w > 0.99
        assert mask.sum() > 3
        testing.assert_allclose(cupy.abs(h[mask]), 0., atol=1e-4)


@testing.with_requires("scipy")
class TestMinimumPhase:

    def test_bad_args(self):
        # not enough taps
        assert_raises(ValueError, signal.minimum_phase, cupy.array([1.]))
        assert_raises(ValueError, signal.minimum_phase, cupy.array([1., 1.]))
        assert_raises(ValueError, signal.minimum_phase, cupy.full(10, 1j))
        assert_raises((AttributeError, ValueError),
                      signal.minimum_phase, 'foo')
        assert_raises(ValueError, signal.minimum_phase, cupy.ones(10), n_fft=8)
        assert_raises(ValueError, signal.minimum_phase,
                      cupy.ones(10), method='foo')

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_homomorphic(self, xp, scp):
        # check that it can recover frequency responses of arbitrary
        # linear-phase filters

        # for some cases we can get the actual filter back
        h = xp.asarray([1, -1])
        h_new = scp.signal.minimum_phase(xp.convolve(h, h[::-1]))
        return h_new

    @pytest.mark.parametrize("n", [2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_homomorphic_2(self, xp, scp, n):
        # but in general we only guarantee we get the magnitude back
        rng = cupy.random.RandomState(0)
        h = rng.randn(n)
        if xp != cupy:
            h = h.get()
        h_new = scp.signal.minimum_phase(xp.convolve(h, h[::-1]))
        return h_new

    @testing.numpy_cupy_allclose(scipy_name="scp", atol=2e-5)
    def test_hilbert(self, xp, scp):  # , n):
        # example from the docstring of `scipy.signal.minimum_phase`
        from scipy.signal import remez
        h_linear = remez(151, [0, 0.2, 0.3, 1.0], [1, 0], fs=2)

        if xp == cupy:
            h_linear = cupy.asarray(h_linear)
        return scp.signal.minimum_phase(h_linear, method="hilbert")

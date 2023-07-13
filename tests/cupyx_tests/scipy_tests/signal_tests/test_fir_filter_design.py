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

    def check_response(self, h, expected_response, tol=.05):
        N = len(h)
        alpha = 0.5 * (N-1)
        m = np.arange(0,N) - alpha   # time indices of taps
        for freq, expected in expected_response:
            actual = abs(np.sum(h*np.exp(-1.j*np.pi*m*freq)))
            mse = abs(actual-expected)**2
            assert_(mse < tol, 'response not as expected, mse=%g > %g'
               % (mse, tol))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_response(self, xp, scp):
        N = 51
        f = .5
        # increase length just to try even/odd
        h = scp.signal.firwin(N, f)  # low-pass from 0 to f
        return h

        self.check_response(h, [(.25,1), (.75,0)])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_response_2(self, xp, scp):
        N = 51
        f = .5
        h = scp.signal.firwin(N+1, f, window='nuttall')  # specific window
        return h

        self.check_response(h, [(.25,1), (.75,0)])

        h = firwin(N+2, f, pass_zero=False)  # stop from 0 to f --> high-pass
        self.check_response(h, [(.25,0), (.75,1)])

        f1, f2, f3, f4 = .2, .4, .6, .8
        h = firwin(N+3, [f1, f2], pass_zero=False)  # band-pass filter
        self.check_response(h, [(.1,0), (.3,1), (.5,0)])

        h = firwin(N+4, [f1, f2])  # band-stop filter
        self.check_response(h, [(.1,1), (.3,0), (.5,1)])

        h = firwin(N+5, [f1, f2, f3, f4], pass_zero=False, scale=False)
        self.check_response(h, [(.1,0), (.3,1), (.5,0), (.7,1), (.9,0)])

        h = firwin(N+6, [f1, f2, f3, f4])  # multiband filter
        self.check_response(h, [(.1,1), (.3,0), (.5,1), (.7,0), (.9,1)])

        h = firwin(N+7, 0.1, width=.03)  # low-pass
        self.check_response(h, [(.05,1), (.75,0)])

        h = firwin(N+8, 0.1, pass_zero=False)  # high-pass
        self.check_response(h, [(.05,0), (.75,1)])

    def mse(self, h, bands):
        """Compute mean squared error versus ideal response across frequency
        band.
          h -- coefficients
          bands -- list of (left, right) tuples relative to 1==Nyquist of
            passbands
        """
        w, H = freqz(h, worN=1024)
        f = w/np.pi
        passIndicator = np.zeros(len(w), bool)
        for left, right in bands:
            passIndicator |= (f >= left) & (f < right)
        Hideal = np.where(passIndicator, 1, 0)
        mse = np.mean(abs(abs(H)-Hideal)**2)
        return mse

    def test_scaling(self):
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
        cases = [
            ([.5], True, (0, 1)),
            ([0.2, .6], False, (.4, 1)),
            ([.5], False, (1, 1)),
        ]
        for cutoff, pass_zero, expected_response in cases:
            h = firwin(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
            hs = firwin(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
            if len(cutoff) == 1:
                if pass_zero:
                    cutoff = [0] + cutoff
                else:
                    cutoff = cutoff + [1]
            assert_(self.mse(h, [cutoff]) < self.mse(hs, [cutoff]),
                'least squares violation')
            self.check_response(hs, [expected_response], 1e-12)

#class TestFirWinMore:
#    """Different author, different style, different tests..."""

    def test_lowpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='lowpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_highpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)

        # Ensure that ntaps is odd.
        ntaps |= 1

        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='highpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_bandpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.2, 0.3-width/2, 0.3+width/2, 0.5,
                                0.7-width/2, 0.7+width/2, 0.8, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='bandpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_bandstop_multi(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta),
                      scale=False)
        taps = firwin(ntaps, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.1, 0.2-width/2, 0.2+width/2, 0.35,
                                0.5-width/2, 0.5+width/2, 0.65,
                                0.8-width/2, 0.8+width/2, 0.9, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                decimal=5)

        taps_str = firwin(ntaps, pass_zero='bandstop', **kwargs)
        assert_allclose(taps, taps_str)

    def test_fs_nyq(self):
        """Test the fs and nyq keywords."""
        nyquist = 1000
        width = 40.0
        relative_width = width/nyquist
        ntaps, beta = kaiserord(120, relative_width)
        taps = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
                        pass_zero=False, scale=False, fs=2*nyquist)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 200, 300-width/2, 300+width/2, 500,
                                700-width/2, 700+width/2, 800, 1000])
        freqs, response = freqz(taps, worN=np.pi*freq_samples/nyquist)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
            taps2 = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
                           pass_zero=False, scale=False, nyq=nyquist)
        assert_allclose(taps2, taps)

    def test_bad_cutoff(self):
        """Test that invalid cutoff argument raises ValueError."""
        # cutoff values must be greater than 0 and less than 1.
        assert_raises(ValueError, firwin, 99, -0.5)
        assert_raises(ValueError, firwin, 99, 1.5)
        # Don't allow 0 or 1 in cutoff.
        assert_raises(ValueError, firwin, 99, [0, 0.5])
        assert_raises(ValueError, firwin, 99, [0.5, 1])
        # cutoff values must be strictly increasing.
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
        # Must have at least one cutoff value.
        assert_raises(ValueError, firwin, 99, [])
        # 2D array not allowed.
        assert_raises(ValueError, firwin, 99, [[0.1, 0.2],[0.3, 0.4]])
        # cutoff values must be less than nyq.
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
            assert_raises(ValueError, firwin, 99, 50.0, nyq=40)
            assert_raises(ValueError, firwin, 99, [10, 20, 30], nyq=25)
        assert_raises(ValueError, firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)

    def test_even_highpass_raises_value_error(self):
        """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
        assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
        assert_raises(ValueError, firwin, 40, [.25, 0.5])

    def test_bad_pass_zero(self):
        """Test degenerate pass_zero cases."""
        with assert_raises(ValueError, match='pass_zero must be'):
            firwin(41, 0.5, pass_zero='foo')
        with assert_raises(TypeError, match='cannot be interpreted'):
            firwin(41, 0.5, pass_zero=1.)
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                firwin(41, [0.5], pass_zero=pass_zero)

    def test_nyq_deprecation(self):
        with pytest.warns(DeprecationWarning,
                          match="Keyword argument 'nyq' is deprecated in "
                          ):
            firwin(1, 1, nyq=10)



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
        taps = scp.signal.firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], [1, 2])
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_2(self, xp, scp):
        # compare to MATLAB output
        taps = scp.signal.firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], [1, 2])
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

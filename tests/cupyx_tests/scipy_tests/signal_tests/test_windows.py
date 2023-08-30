
import warnings

from cupy import testing
import cupyx.scipy.signal.windows as cu_windows

import pytest
from pytest import raises as assert_raises


try:
    import scipy.signal.windows as cpu_windows  # NOQA
    import scipy.fft  # NOQA
except ImportError:
    pass


window_funcs = [
    ('boxcar', ()),
    ('triang', ()),
    ('parzen', ()),
    ('bohman', ()),
    ('blackman', ()),
    ('nuttall', ()),
    ('blackmanharris', ()),
    ('flattop', ()),
    ('bartlett', ()),
    ('barthann', ()),
    ('hamming', ()),
    ('kaiser', (1,)),
    ('gaussian', (0.5,)),
    ('general_gaussian', (1.5, 2)),
    ('chebwin', (1,)),
    ('cosine', ()),
    ('hann', ()),
    ('exponential', ()),
    ('taylor', ()),
    ('tukey', (0.5,)),
]


class TestBartHann:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15, atol=1e-15)
    def test_basic(self, xp, scp):
        w1 = scp.signal.windows.barthann(6, sym=True)
        w2 = scp.signal.windows.barthann(7)
        w3 = scp.signal.windows.barthann(6, False)
        return w1, w2, w3


class TestBartlett:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15, atol=1e-15)
    def test_basic(self, xp, scp):
        w1 = scp.signal.windows.bartlett(6)
        w2 = scp.signal.windows.bartlett(7)
        w3 = scp.signal.windows.bartlett(6, False)
        return w1, w2, w3


class TestBlackman:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15, atol=1e-15)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.blackman(6, sym=False),
                scp.signal.windows.blackman(7, sym=False),
                scp.signal.windows.blackman(6),
                scp.signal.windows.blackman(7, True))


class TestBlackmanHarris:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15, atol=1e-15)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.blackmanharris(6, False),
                scp.signal.windows.blackmanharris(7, sym=False),
                scp.signal.windows.blackmanharris(6),
                scp.signal.windows.blackmanharris(7, sym=True))


class TestTaylor:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15, atol=1e-15)
    def test_normalized(self, xp, scp):
        """Tests windows of small length that are normalized to 1. See the
        documentation for the Taylor window for more information on
        normalization.
        """
        w1 = scp.signal.windows.taylor(1, 2, 15)
        w2 = scp.signal.windows.taylor(6, 2, 15)
        return w1, w2

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15, atol=1e-15)
    def test_non_normalized(self, xp, scp):
        """Test windows of small length that are not normalized to 1. See
        the documentation for the Taylor window for more information on
        normalization.
        """
        return (scp.signal.windows.taylor(5, 2, 15, norm=False),
                scp.signal.windows.taylor(6, 2, 15, norm=False))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_correctness(self, xp, scp):
        """This test ensures the correctness of the implemented Taylor
        Windowing function. A Taylor Window of 1024 points is created, its FFT
        is taken, and the Peak Sidelobe Level (PSLL) and 3dB and 18dB bandwidth
        are found and checked.

        A publication from Sandia National Laboratories was used as reference
        for the correctness values [1]_.

        References
        -----
        .. [1] Armin Doerry, "Catalog of Window Taper Functions for
               Sidelobe Control", 2017.
               https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
        """
        M_win = 1024
        N_fft = 131072
        # Set norm=False for correctness as the values obtained from the
        # scientific publication do not normalize the values. Normalizing
        # changes the sidelobe level from the desired value.
        w = scp.signal.windows.taylor(
            M_win, nbar=4, sll=35, norm=False, sym=False)
        f = scp.fft.fft(w, N_fft)
        spec = 20 * xp.log10(xp.abs(f / xp.amax(f)))

        first_zero = xp.argmax(xp.diff(spec) > 0)

        PSLL = xp.amax(spec[first_zero:-first_zero])

        BW_3dB = 2 * xp.argmax(spec <= -3.0102999566398121) / N_fft * M_win
        BW_18dB = 2 * xp.argmax(spec <= -18.061799739838872) / N_fft * M_win

        return PSLL, BW_3dB, BW_18dB


class TestBohman:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.bohman(6),
                scp.signal.windows.bohman(7, sym=True),
                scp.signal.windows.bohman(6, False))


class TestBoxcar:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.boxcar(6),
                scp.signal.windows.boxcar(7),
                scp.signal.windows.boxcar(6, False))


class TestChebWin:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            ret = (scp.signal.windows.chebwin(6, 100),
                   scp.signal.windows.chebwin(7, 100),
                   scp.signal.windows.chebwin(6, 10),
                   scp.signal.windows.chebwin(7, 10),
                   scp.signal.windows.chebwin(6, 10, False))
        return ret

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_cheb_odd_high_attenuation(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            cheb_odd = scp.signal.windows.chebwin(53, at=-40)
        return cheb_odd

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_cheb_even_high_attenuation(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            cheb_even = scp.signal.windows.chebwin(54, at=40)
        return cheb_even

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_cheb_odd_low_attenuation(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            cheb_odd = scp.signal.windows.chebwin(7, at=10)
        return cheb_odd

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_cheb_even_low_attenuation(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            cheb_even = scp.signal.windows.chebwin(8, at=-10)
        return cheb_even


exponential_data = {
    (4, None, 0.2, False): True,
    (4, None, 0.2, True): True,
    (4, None, 1.0, False): True,
    (4, None, 1.0, True): True,
    (4, 2, 0.2, False): True,
    (4, 2, 0.2, True): False,
    (4, 2, 1.0, False): True,
    (4, 2, 1.0, True): False,
    (5, None, 0.2, True): True,
    (5, None, 1.0, True): True,
    (5, 2, 0.2, True): False,
    (5, 2, 1.0, True): False
}


@testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
def test_exponential(xp, scp):
    for args, valid in exponential_data.items():
        if not valid:
            assert_raises(ValueError, scp.signal.windows.exponential, *args)
        else:
            win = scp.signal.windows.exponential(*args)
            return win


class TestFlatTop:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.flattop(6, sym=False),
                scp.signal.windows.flattop(7, sym=False),
                scp.signal.windows.flattop(6),
                scp.signal.windows.flattop(7, True),)


class TestGaussian:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.gaussian(6, 1.0),
                scp.signal.windows.gaussian(7, 1.2),
                scp.signal.windows.gaussian(7, 3),
                scp.signal.windows.gaussian(6, 3, False),)


class TestGeneralCosine:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.general_cosine(5, [0.5, 0.3, 0.2]),
                scp.signal.windows.general_cosine(4, [0.5, 0.3, 0.2],
                                                  sym=False),)


class TestGeneralHamming:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.general_hamming(5, 0.7),
                scp.signal.windows.general_hamming(5, 0.75, sym=False),
                scp.signal.windows.general_hamming(6, 0.75, sym=True),)


class TestHamming:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.hamming(6, False),
                scp.signal.windows.hamming(7, sym=False),
                scp.signal.windows.hamming(6),
                scp.signal.windows.hamming(7, sym=True),)


class TestHann:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.hann(6, sym=False),
                scp.signal.windows.hann(7, sym=False),
                scp.signal.windows.hann(6, True),
                scp.signal.windows.hann(7),)


class TestKaiser:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.kaiser(6, 0.5),
                scp.signal.windows.kaiser(7, 0.5),
                scp.signal.windows.kaiser(6, 2.7),
                scp.signal.windows.kaiser(7, 2.7),
                scp.signal.windows.kaiser(6, 2.7, False),)


@pytest.mark.skip('This has not been implemented yet in CuPy')
class TestKaiserBesselDerived:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        M = 100
        w = scp.signal.windows.kaiser_bessel_derived(M, beta=4.0)
        w2 = scp.signal.windows.get_window(
            ('kaiser bessel derived', 4.0), M, fftbins=False)
        w3 = scp.signal.windows.kaiser_bessel_derived(2, beta=xp.pi / 2)
        w4 = scp.signal.windows.kaiser_bessel_derived(4, beta=xp.pi / 2)
        w5 = scp.signal.windows.kaiser_bessel_derived(6, beta=xp.pi / 2)
        return w, w2, w3, w4, w5

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_exceptions(self, xp, scp):
        M = 100
        # Assert ValueError for odd window length
        msg = ("Kaiser-Bessel Derived windows are only defined for even "
               "number of points")
        with assert_raises(ValueError, match=msg):
            scp.signal.windows.kaiser_bessel_derived(M + 1, beta=4.)

        # Assert ValueError for non-symmetric setting
        msg = ("Kaiser-Bessel Derived windows are only defined for "
               "symmetric shapes")
        with assert_raises(ValueError, match=msg):
            scp.signal.windows.kaiser_bessel_derived(M + 1, beta=4., sym=False)


class TestNuttall:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.nuttall(6, sym=False),
                scp.signal.windows.nuttall(7, sym=False),
                scp.signal.windows.nuttall(6),
                scp.signal.windows.nuttall(7, True),)


class TestParzen:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.parzen(6),
                scp.signal.windows.parzen(7, sym=True),
                scp.signal.windows.parzen(6, False),)


class TestTriang:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        return (scp.signal.windows.triang(6, True),
                scp.signal.windows.triang(7),
                scp.signal.windows.triang(6, sym=False),)


tukey_data = [
    (4, 0.5, True),
    (4, 0.9, True),
    (4, 1.0, True),
    (4, 0.5, False),
    (4, 0.9, False),
    (4, 1.0, False),
    (5, 0.0, True),
    (5, 0.8, True),
    (5, 1.0, True),
    (6, 0),
    (7, 0),
    (6, .25),
    (7, .25),
    (6,),
    (7,),
    (6, .75),
    (7, .75),
    (6, 1),
    (7, 1),
]


class TestTukey:
    @pytest.mark.parametrize('args', tukey_data)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, args, xp, scp):
        # Test against hardcoded data
        win = scp.signal.windows.tukey(*args)
        return win


dpss_data = [
    (4, 0.1, 2),
    (3, 1.4, 3),
    (5, 1.5, 5),
    (100, 2, 4),
]


@pytest.mark.skip('This has not been implemented yet in CuPy')
class TestDPSS:
    @pytest.mark.parametrize('args', tukey_data)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, args, xp, scp):
        win, ratios = scp.signal.windows.dpss(*args, return_ratios=True)
        return win, ratios

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_unity(self, xp, scp):
        # Test unity value handling (gh-2221)
        results = []
        for M in range(1, 21):
            # corrected w/approximation (default)
            win = scp.signal.windows.dpss(M, M / 2.1)
            results.append(win)
            # corrected w/subsample delay (slower)
            win_sub = scp.signal.windows.dpss(M, M / 2.1, norm='subsample')
            if M > 2:
                # @M=2 the subsample doesn't do anything
                results.append(win_sub)
            # not the same, l2-norm
            win_2 = scp.signal.windows.dpss(M, M / 2.1, norm=2)
            results.append(win_2)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_extremes(self, xp, scp):
        # Test extremes of alpha
        lam1 = scp.signal.windows.dpss(31, 6, 4, return_ratios=True)[1]
        lam2 = scp.signal.windows.dpss(31, 7, 4, return_ratios=True)[1]
        lam3 = scp.signal.windows.dpss(31, 8, 4, return_ratios=True)[1]
        return lam1, lam2, lam3

    @pytest.mark.parametrize('windows', [cu_windows, cpu_windows])
    def test_degenerate(self, windows):
        # Test failures
        assert_raises(ValueError, windows.dpss, 4, 1.5, -1)  # Bad Kmax
        assert_raises(ValueError, windows.dpss, 4, 1.5, -5)
        assert_raises(TypeError, windows.dpss, 4, 1.5, 1.1)
        assert_raises(ValueError, windows.dpss, 3, 1.5, 3)  # NW must be < N/2.
        assert_raises(ValueError, windows.dpss, 3, -1, 3)  # NW must be pos
        assert_raises(ValueError, windows.dpss, 3, 0, 3)
        assert_raises(ValueError, windows.dpss, -1, 1, 3)  # negative M


@pytest.mark.skip('This has not been implemented yet in CuPy')
class TestLanczos:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_basic(self, xp, scp):
        # Analytical results:
        # sinc(x) = sinc(-x)
        # sinc(pi) = 0, sinc(0) = 1
        # Hand computation on WolframAlpha:
        # sinc(2 pi / 3) = 0.413496672
        # sinc(pi / 3) = 0.826993343
        # sinc(3 pi / 5) = 0.504551152
        # sinc(pi / 5) = 0.935489284
        return (scp.signal.windows.lanczos(6, sym=False),
                scp.signal.windows.lanczos(6),
                scp.signal.windows.lanczos(7, sym=True),)

    @pytest.mark.parametrize('windows', [cu_windows, cpu_windows])
    def test_array_size(self, windows):
        for n in [0, 10, 11]:
            assert len(windows.lanczos(n, sym=False)) == n
            assert len(windows.lanczos(n, sym=True)) == n


class TestGetWindow:
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_boxcar(self, xp, scp):
        w1 = scp.signal.windows.get_window('boxcar', 12)

        # window is a tuple of len 1
        w2 = scp.signal.windows.get_window(('boxcar',), 16)
        return w1, w2

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_cheb_odd(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            w = scp.signal.windows.get_window(
                ('chebwin', -40), 53, fftbins=False)
        return w

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_cheb_even(self, xp, scp):
        with warnings.catch_warnings():
            # sup.filter(UserWarning, "This window is not suitable")
            w = scp.signal.windows.get_window(
                ('chebwin', 40), 54, fftbins=False)
        return w

    @pytest.mark.skip('This has not been implemented yet in CuPy')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_dpss(self, xp, scp):
        win1 = scp.signal.windows.get_window(('dpss', 3), 64, fftbins=False)
        win2 = scp.signal.windows.dpss(64, 3)
        return win1, win2

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_kaiser_float(self, xp, scp):
        win1 = scp.signal.windows.get_window(7.2, 64)
        win2 = scp.signal.windows.kaiser(64, 7.2, False)
        return win1, win2

    @pytest.mark.parametrize('windows', [cu_windows, cpu_windows])
    def test_invalid_inputs(self, windows):
        # Window is not a float, tuple, or string
        assert_raises(ValueError, windows.get_window, set('hann'), 8)

        # Unknown window type error
        assert_raises(ValueError, windows.get_window, 'broken', 4)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_array_as_window(self, xp, scp):
        # scipy github issue 3603
        osfactor = 128
        sig = xp.arange(128)

        win = scp.signal.windows.get_window(('kaiser', 8.0), osfactor // 2)
        if hasattr(scp.signal, 'resample'):
            with assert_raises(ValueError, match='must have the same length'):
                scp.signal.resample(sig, len(sig) * osfactor, window=win)
        return win

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_general_cosine(self, xp, scp):
        return (scp.signal.get_window(('general_cosine', [0.5, 0.3, 0.2]), 4),
                scp.signal.get_window(('general_cosine', [0.5, 0.3, 0.2]), 4,
                                      fftbins=False))

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
    def test_general_hamming(self, xp, scp):
        return (
            scp.signal.get_window(('general_hamming', 0.7), 5),
            scp.signal.get_window(('general_hamming', 0.7), 5, fftbins=False),)

    @pytest.mark.skip('This has not been implemented yet in CuPy')
    def test_lanczos(self, xp, scp):
        return (scp.signal.get_window('lanczos', 6),
                scp.signal.get_window('lanczos', 6, fftbins=False),
                scp.signal.get_window('lanczos', 6),
                scp.signal.get_window('sinc', 6))


@pytest.mark.parametrize('window_info', window_funcs)
@testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
def test_windowfunc_basics(window_info, xp, scp):
    window_name, params = window_info
    if window_name in {'parzen', 'tukey'}:
        pytest.skip()

    window = getattr(scp.signal.windows, window_name)
    results = []
    with warnings.catch_warnings():
        # Check symmetry for odd and even lengths
        w1 = window(8, *params, sym=True)
        w2 = window(7, *params, sym=False)
        results += [w1, w2]

        w1 = window(9, *params, sym=True)
        w2 = window(8, *params, sym=False)
        results += [w1, w2]

        # Check that functions run and output lengths are correct
        results.append(len(window(6, *params, sym=True)))
        results.append(len(window(6, *params, sym=False)))
        results.append(len(window(7, *params, sym=True)))
        results.append(len(window(7, *params, sym=False)))

        # Check invalid lengths
        assert_raises((ValueError, TypeError), window, 5.5, *params)
        assert_raises((ValueError, TypeError), window, -7, *params)

        # Check degenerate cases
        results.append(window(0, *params, sym=True))
        results.append(window(0, *params, sym=False))
        results.append(window(1, *params, sym=True))
        results.append(window(1, *params, sym=False))

        # Check normalization
        results.append(window(10, *params, sym=True))
        results.append(window(10, *params, sym=False))
        results.append(window(9, *params, sym=True))
        results.append(window(9, *params, sym=False))

        # Check that DFT-even spectrum is purely real for odd and even
        results.append(scp.fft.fft(window(10, *params, sym=False)).imag)
        results.append(scp.fft.fft(window(11, *params, sym=False)).imag)

    return results


@pytest.mark.parametrize('windows', [cu_windows, cpu_windows])
def test_needs_params(windows):
    for winstr in ['kaiser', 'ksr', 'kaiser_bessel_derived', 'kbd',
                   'gaussian', 'gauss', 'gss',
                   'general gaussian', 'general_gaussian',
                   'general gauss', 'general_gauss', 'ggs',
                   'dss', 'dpss', 'general cosine', 'general_cosine',
                   'chebwin', 'cheb', 'general hamming', 'general_hamming',
                   ]:
        assert_raises(ValueError, windows.get_window, winstr, 7)


@testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13, atol=1e-13)
def test_not_needs_params(xp, scp):
    for winstr in ['barthann',
                   'bartlett',
                   'blackman',
                   'blackmanharris',
                   'bohman',
                   'boxcar',
                   'cosine',
                   'flattop',
                   'hamming',
                   'nuttall',
                   'parzen',
                   'taylor',
                   'exponential',
                   'poisson',
                   'tukey',
                   'tuk',
                   'triangle',
                   ]:
        win = scp.signal.get_window(winstr, 7)
        return win

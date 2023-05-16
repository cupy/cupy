
import warnings

# import cupy
from cupy import testing
# import cupyx.scipy.signal.windows as windows
# from cupyx.scipy.signal.windows import get_window

# import numpy as np
import pytest
# from cupy import array
from pytest import raises as assert_raises

# from cupyx.scipy.fft import fft

try:
    import scipy.signal.windows  # NOQA
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


@pytest.mark.skip('The kernel requires a detailed inspection against SciPy')
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


dpss_data = {
    # All values from MATLAB:
    # * taper[1] of (3, 1.4, 3) sign-flipped
    # * taper[3] of (5, 1.5, 5) sign-flipped
    (4, 0.1, 2): ([[0.497943898, 0.502047681, 0.502047681, 0.497943898], [0.670487993, 0.224601537, -0.224601537, -0.670487993]], [0.197961815, 0.002035474]),  # noqa
    (3, 1.4, 3): ([[0.410233151, 0.814504464, 0.410233151], [0.707106781, 0.0, -0.707106781], [0.575941629, -0.580157287, 0.575941629]], [0.999998093, 0.998067480, 0.801934426]),  # noqa
    (5, 1.5, 5): ([[0.1745071052, 0.4956749177, 0.669109327, 0.495674917, 0.174507105], [0.4399493348, 0.553574369, 0.0, -0.553574369, -0.439949334], [0.631452756, 0.073280238, -0.437943884, 0.073280238, 0.631452756], [0.553574369, -0.439949334, 0.0, 0.439949334, -0.553574369], [0.266110290, -0.498935248, 0.600414741, -0.498935248, 0.266110290147157]], [0.999728571, 0.983706916, 0.768457889, 0.234159338, 0.013947282907567]),  # noqa: E501
    (100, 2, 4): ([[0.0030914414, 0.0041266922, 0.005315076, 0.006665149, 0.008184854, 0.0098814158, 0.011761239, 0.013829809, 0.016091597, 0.018549973, 0.02120712, 0.02406396, 0.027120092, 0.030373728, 0.033821651, 0.037459181, 0.041280145, 0.045276872, 0.049440192, 0.053759447, 0.058222524, 0.062815894, 0.067524661, 0.072332638, 0.077222418, 0.082175473, 0.087172252, 0.092192299, 0.097214376, 0.1022166, 0.10717657, 0.11207154, 0.11687856, 0.12157463, 0.12613686, 0.13054266, 0.13476986, 0.13879691, 0.14260302, 0.14616832, 0.14947401, 0.1525025, 0.15523755, 0.15766438, 0.15976981, 0.16154233, 0.16297223, 0.16405162, 0.16477455, 0.16513702, 0.16513702, 0.16477455, 0.16405162, 0.16297223, 0.16154233, 0.15976981, 0.15766438, 0.15523755, 0.1525025, 0.14947401, 0.14616832, 0.14260302, 0.13879691, 0.13476986, 0.13054266, 0.12613686, 0.12157463, 0.11687856, 0.11207154, 0.10717657, 0.1022166, 0.097214376, 0.092192299, 0.087172252, 0.082175473, 0.077222418, 0.072332638, 0.067524661, 0.062815894, 0.058222524, 0.053759447, 0.049440192, 0.045276872, 0.041280145, 0.037459181, 0.033821651, 0.030373728, 0.027120092, 0.02406396, 0.02120712, 0.018549973, 0.016091597, 0.013829809, 0.011761239, 0.0098814158, 0.008184854, 0.006665149, 0.005315076, 0.0041266922, 0.0030914414], [0.018064449, 0.022040342, 0.026325013, 0.030905288, 0.035764398, 0.040881982, 0.046234148, 0.051793558, 0.057529559, 0.063408356, 0.069393216, 0.075444716, 0.081521022, 0.087578202, 0.093570567, 0.099451049, 0.10517159, 0.11068356, 0.11593818, 0.12088699, 0.12548227, 0.12967752, 0.1334279, 0.13669069, 0.13942569, 0.1415957, 0.14316686, 0.14410905, 0.14439626, 0.14400686, 0.14292389, 0.1411353, 0.13863416, 0.13541876, 0.13149274, 0.12686516, 0.12155045, 0.1155684, 0.10894403, 0.10170748, 0.093893752, 0.08554251, 0.076697768, 0.067407559, 0.057723559, 0.04770068, 0.037396627, 0.026871428, 0.016186944, 0.0054063557, -0.0054063557, -0.016186944, -0.026871428, -0.037396627, -0.04770068, -0.057723559, -0.067407559, -0.076697768, -0.08554251, -0.093893752, -0.10170748, -0.10894403, -0.1155684, -0.12155045, -0.12686516, -0.13149274, -0.13541876, -0.13863416, -0.1411353, -0.14292389, -0.14400686, -0.14439626, -0.14410905, -0.14316686, -0.1415957, -0.13942569, -0.13669069, -0.1334279, -0.12967752, -0.12548227, -0.12088699, -0.11593818, -0.11068356, -0.10517159, -0.099451049, -0.093570567, -0.087578202, -0.081521022, -0.075444716, -0.069393216, -0.063408356, -0.057529559, -0.051793558, -0.046234148, -0.040881982, -0.035764398, -0.030905288, -0.026325013, -0.022040342, -0.018064449], [0.064817553, 0.072567801, 0.080292992, 0.087918235, 0.095367076, 0.10256232, 0.10942687, 0.1158846, 0.12186124, 0.12728523, 0.13208858, 0.13620771, 0.13958427, 0.14216587, 0.14390678, 0.14476863, 0.1447209, 0.14374148, 0.14181704, 0.13894336, 0.13512554, 0.13037812, 0.1247251, 0.11819984, 0.11084487, 0.10271159, 0.093859853, 0.084357497, 0.074279719, 0.063708406, 0.052731374, 0.041441525, 0.029935953, 0.018314987, 0.0066811877, -0.0048616765, -0.016209689, -0.027259848, -0.037911124, -0.048065512, -0.05762905, -0.066512804, -0.0746338, -0.081915903, -0.088290621, -0.09369783, -0.098086416, -0.10141482, -0.10365146, -0.10477512, -0.10477512, -0.10365146, -0.10141482, -0.098086416, -0.09369783, -0.088290621, -0.081915903, -0.0746338, -0.066512804, -0.05762905, -0.048065512, -0.037911124, -0.027259848, -0.016209689, -0.0048616765, 0.0066811877, 0.018314987, 0.029935953, 0.041441525, 0.052731374, 0.063708406, 0.074279719, 0.084357497, 0.093859853, 0.10271159, 0.11084487, 0.11819984, 0.1247251, 0.13037812, 0.13512554, 0.13894336, 0.14181704, 0.14374148, 0.1447209, 0.14476863, 0.14390678, 0.14216587, 0.13958427, 0.13620771, 0.13208858, 0.12728523, 0.12186124, 0.1158846, 0.10942687, 0.10256232, 0.095367076, 0.087918235, 0.080292992, 0.072567801, 0.064817553], [0.14985551, 0.15512305, 0.15931467, 0.16236806, 0.16423291, 0.16487165, 0.16426009, 0.1623879, 0.1592589, 0.15489114, 0.14931693, 0.14258255, 0.13474785, 0.1258857, 0.11608124, 0.10543095, 0.094041635, 0.082029213, 0.069517411, 0.056636348, 0.043521028, 0.030309756, 0.017142511, 0.0041592774, -0.0085016282, -0.020705223, -0.032321494, -0.043226982, -0.053306291, -0.062453515, -0.070573544, -0.077583253, -0.083412547, -0.088005244, -0.091319802, -0.093329861, -0.094024602, -0.093408915, -0.091503383, -0.08834406, -0.08398207, -0.078483012, -0.071926192, -0.064403681, -0.056019215, -0.046886954, -0.037130106, -0.026879442, -0.016271713, -0.005448, 0.005448, 0.016271713, 0.026879442, 0.037130106, 0.046886954, 0.056019215, 0.064403681, 0.071926192, 0.078483012, 0.08398207, 0.08834406, 0.091503383, 0.093408915, 0.094024602, 0.093329861, 0.091319802, 0.088005244, 0.083412547, 0.077583253, 0.070573544, 0.062453515, 0.053306291, 0.043226982, 0.032321494, 0.020705223, 0.0085016282, -0.0041592774, -0.017142511, -0.030309756, -0.043521028, -0.056636348, -0.069517411, -0.082029213, -0.094041635, -0.10543095, -0.11608124, -0.1258857, -0.13474785, -0.14258255, -0.14931693, -0.15489114, -0.1592589, -0.1623879, -0.16426009, -0.16487165, -0.16423291, -0.16236806, -0.15931467, -0.15512305, -0.14985551]], [0.999943140, 0.997571533, 0.959465463, 0.721862496]),  # noqa: E501
}

"""
class TestDPSS:

    def test_basic(self):
        # Test against hardcoded data
        for k, v in dpss_data.items():
            win, ratios = windows.dpss(*k, return_ratios=True)
            assert_allclose(win, v[0], atol=1e-7, err_msg=k)
            assert_allclose(ratios, v[1], rtol=1e-5, atol=1e-7, err_msg=k)

    def test_unity(self):
        # Test unity value handling (gh-2221)
        for M in range(1, 21):
            # corrected w/approximation (default)
            win = windows.dpss(M, M / 2.1)
            expected = M % 2  # one for odd, none for even
            assert_equal(np.isclose(win, 1.).sum(), expected,
                         err_msg=f'{win}')
            # corrected w/subsample delay (slower)
            win_sub = windows.dpss(M, M / 2.1, norm='subsample')
            if M > 2:
                # @M=2 the subsample doesn't do anything
                assert_equal(np.isclose(win_sub, 1.).sum(), expected,
                             err_msg=f'{win_sub}')
                assert_allclose(win, win_sub, rtol=0.03)  # within 3%
            # not the same, l2-norm
            win_2 = windows.dpss(M, M / 2.1, norm=2)
            expected = 1 if M == 1 else 0
            assert_equal(np.isclose(win_2, 1.).sum(), expected,
                         err_msg=f'{win_2}')

    def test_extremes(self):
        # Test extremes of alpha
        lam = windows.dpss(31, 6, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)
        lam = windows.dpss(31, 7, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)
        lam = windows.dpss(31, 8, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)

    def test_degenerate(self):
        # Test failures
        assert_raises(ValueError, windows.dpss, 4, 1.5, -1)  # Bad Kmax
        assert_raises(ValueError, windows.dpss, 4, 1.5, -5)
        assert_raises(TypeError, windows.dpss, 4, 1.5, 1.1)
        assert_raises(ValueError, windows.dpss, 3, 1.5, 3)  # NW must be < N/2.
        assert_raises(ValueError, windows.dpss, 3, -1, 3)  # NW must be pos
        assert_raises(ValueError, windows.dpss, 3, 0, 3)
        assert_raises(ValueError, windows.dpss, -1, 1, 3)  # negative M


class TestLanczos:

    def test_basic(self):
        # Analytical results:
        # sinc(x) = sinc(-x)
        # sinc(pi) = 0, sinc(0) = 1
        # Hand computation on WolframAlpha:
        # sinc(2 pi / 3) = 0.413496672
        # sinc(pi / 3) = 0.826993343
        # sinc(3 pi / 5) = 0.504551152
        # sinc(pi / 5) = 0.935489284
        assert_allclose(windows.lanczos(6, sym=False),
                        [0., 0.413496672,
                         0.826993343, 1., 0.826993343,
                         0.413496672],
                        atol=1e-9)
        assert_allclose(windows.lanczos(6),
                        [0., 0.504551152,
                         0.935489284, 0.935489284,
                         0.504551152, 0.],
                        atol=1e-9)
        assert_allclose(windows.lanczos(7, sym=True),
                        [0., 0.413496672,
                         0.826993343, 1., 0.826993343,
                         0.413496672, 0.],
                        atol=1e-9)

    def test_array_size(self):
        for n in [0, 10, 11]:
            assert_equal(len(windows.lanczos(n, sym=False)), n)
            assert_equal(len(windows.lanczos(n, sym=True)), n)


class TestGetWindow:

    def test_boxcar(self):
        w = windows.get_window('boxcar', 12)
        assert_array_equal(w, np.ones_like(w))

        # window is a tuple of len 1
        w = windows.get_window(('boxcar',), 16)
        assert_array_equal(w, np.ones_like(w))

    def test_cheb_odd(self):
        with suppress_warnings():
            sup.filter(UserWarning, "This window is not suitable")
            w = windows.get_window(('chebwin', -40), 53, fftbins=False)
        assert_array_almost_equal(w, cheb_odd_true, decimal=4)

    def test_cheb_even(self):
        with suppress_warnings():
            sup.filter(UserWarning, "This window is not suitable")
            w = windows.get_window(('chebwin', 40), 54, fftbins=False)
        assert_array_almost_equal(w, cheb_even_true, decimal=4)

    def test_dpss(self):
        win1 = windows.get_window(('dpss', 3), 64, fftbins=False)
        win2 = windows.dpss(64, 3)
        assert_array_almost_equal(win1, win2, decimal=4)

    def test_kaiser_float(self):
        win1 = windows.get_window(7.2, 64)
        win2 = windows.kaiser(64, 7.2, False)
        assert_allclose(win1, win2)

    def test_invalid_inputs(self):
        # Window is not a float, tuple, or string
        assert_raises(ValueError, windows.get_window, set('hann'), 8)

        # Unknown window type error
        assert_raises(ValueError, windows.get_window, 'broken', 4)

    def test_array_as_window(self):
        # github issue 3603
        osfactor = 128
        sig = np.arange(128)

        win = windows.get_window(('kaiser', 8.0), osfactor // 2)
        with assert_raises(ValueError, match='must have the same length'):
            resample(sig, len(sig) * osfactor, window=win)

    def test_general_cosine(self):
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4),
                        [0.4, 0.3, 1, 0.3])
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4,
                                   fftbins=False),
                        [0.4, 0.55, 0.55, 0.4])

    def test_general_hamming(self):
        assert_allclose(get_window(('general_hamming', 0.7), 5),
                        [0.4, 0.6072949, 0.9427051, 0.9427051, 0.6072949])
        assert_allclose(get_window(('general_hamming', 0.7), 5, fftbins=False),
                        [0.4, 0.7, 1.0, 0.7, 0.4])

    def test_lanczos(self):
        assert_allclose(get_window('lanczos', 6),
                        [0., 0.413496672, 0.826993343, 1., 0.826993343,
                         0.413496672], atol=1e-9)
        assert_allclose(get_window('lanczos', 6, fftbins=False),
                        [0., 0.504551152, 0.935489284, 0.935489284,
                         0.504551152, 0.], atol=1e-9)
        assert_allclose(get_window('lanczos', 6), get_window('sinc', 6))


def test_windowfunc_basics():
    for window_name, params in window_funcs:
        window = getattr(windows, window_name)
        with suppress_warnings():
            sup.filter(UserWarning, "This window is not suitable")
            # Check symmetry for odd and even lengths
            w1 = window(8, *params, sym=True)
            w2 = window(7, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            w1 = window(9, *params, sym=True)
            w2 = window(8, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            # Check that functions run and output lengths are correct
            assert_equal(len(window(6, *params, sym=True)), 6)
            assert_equal(len(window(6, *params, sym=False)), 6)
            assert_equal(len(window(7, *params, sym=True)), 7)
            assert_equal(len(window(7, *params, sym=False)), 7)

            # Check invalid lengths
            assert_raises(ValueError, window, 5.5, *params)
            assert_raises(ValueError, window, -7, *params)

            # Check degenerate cases
            assert_array_equal(window(0, *params, sym=True), [])
            assert_array_equal(window(0, *params, sym=False), [])
            assert_array_equal(window(1, *params, sym=True), [1])
            assert_array_equal(window(1, *params, sym=False), [1])

            # Check dtype
            assert_(window(0, *params, sym=True).dtype == 'float')
            assert_(window(0, *params, sym=False).dtype == 'float')
            assert_(window(1, *params, sym=True).dtype == 'float')
            assert_(window(1, *params, sym=False).dtype == 'float')
            assert_(window(6, *params, sym=True).dtype == 'float')
            assert_(window(6, *params, sym=False).dtype == 'float')

            # Check normalization
            assert_array_less(window(10, *params, sym=True), 1.01)
            assert_array_less(window(10, *params, sym=False), 1.01)
            assert_array_less(window(9, *params, sym=True), 1.01)
            assert_array_less(window(9, *params, sym=False), 1.01)

            # Check that DFT-even spectrum is purely real for odd and even
            assert_allclose(fft(window(10, *params, sym=False)).imag,
                            0, atol=1e-14)
            assert_allclose(fft(window(11, *params, sym=False)).imag,
                            0, atol=1e-14)


def test_needs_params():
    for winstr in ['kaiser', 'ksr', 'kaiser_bessel_derived', 'kbd',
                   'gaussian', 'gauss', 'gss',
                   'general gaussian', 'general_gaussian',
                   'general gauss', 'general_gauss', 'ggs',
                   'dss', 'dpss', 'general cosine', 'general_cosine',
                   'chebwin', 'cheb', 'general hamming', 'general_hamming',
                   ]:
        assert_raises(ValueError, get_window, winstr, 7)


def test_not_needs_params():
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
        win = get_window(winstr, 7)
        assert_equal(len(win), 7)
"""

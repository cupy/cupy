import cupy

import cupyx.scipy.signal as signal
from cupy import testing

import pytest
from pytest import raises as assert_raises


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

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
        assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
        # negative desired
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
        # len(weight) != len(pairs)
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [1, 2])
        # negative weight
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [-1])

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-13)
    def test_firls(self, xp, scp):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band
        # design a halfband symmetric low-pass filter
        h = scp.signal.firls(11, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)
        return h

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_firls_freqz(self, xp, scp):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band

        # design a halfband symmetric low-pass filter and check the frequency response
        h = scp.signal.firls(11, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)
        w, H = scp.signal.freqz(h, 1)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare(self, xp, scp):
        # compare to OCTAVE output
        taps = scp.signal.firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], [1, 2])
        return taps

  #      # >> taps = firls(8, [0 0.5 0.55 1], [1 1 0 0], [1, 2]);
  #      known_taps = [-6.26930101730182e-04, -1.03354450635036e-01,
  #                    -9.81576747564301e-03, 3.17271686090449e-01,
  #                    5.11409425599933e-01, 3.17271686090449e-01,
  #                    -9.81576747564301e-03, -1.03354450635036e-01,
  #                    -6.26930101730182e-04]
  #      assert_allclose(taps, known_taps)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_2(self, xp, scp):
        # compare to MATLAB output
        taps = scp.signal.firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], [1, 2])
        return taps

        # >> taps = firls(10, [0 0.5 0.5 1], [1 1 0 0], [1, 2]);
 #       known_taps = [
 #           0.058545300496815, -0.014233383714318, -0.104688258464392,
 #           0.012403323025279, 0.317930861136062, 0.488047220029700,
 #           0.317930861136062, 0.012403323025279, -0.104688258464392,
 #           -0.014233383714318, 0.058545300496815]
 #       assert_allclose(taps, known_taps)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_3(self, xp, scp):
        # With linear changes:
        taps = scp.signal.firls(7, (0, 1, 2, 3, 4, 5), [
                                1, 0, 0, 1, 1, 0], fs=20)
        return taps

        # >> taps = firls(6, [0, 0.1, 0.2, 0.3, 0.4, 0.5], [1, 0, 0, 1, 1, 0])
 #       known_taps = [
 #           1.156090832768218, -4.1385894727395849, 7.5288619164321826,
 #           -8.5530572592947856, 7.5288619164321826, -4.1385894727395849,
 #           1.156090832768218]
 #       assert_allclose(taps, known_taps)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rank_deficient(self, xp, scp):
        # solve() runs but warns (only sometimes, so here we don't use match)
        x = scp.signal.firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        w, h = scp.signal.freqz(x, fs=2.)
        return x, w, h

#        assert_allclose(np.abs(h[:2]), 1., atol=1e-5)
#        assert_allclose(np.abs(h[-2:]), 0., atol=1e-6)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rank_deficient_2(self, xp, scp):
        # switch to pinvh (tolerances could be higher with longer
        # filters, but using shorter ones is faster computationally and
        # the idea is the same)
        x = scp.signal.firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = scp.signal.freqz(x, fs=2.)
        return x, w, h

#        mask = w < 0.01
#        assert mask.sum() > 3
#        assert_allclose(np.abs(h[mask]), 1., atol=1e-4)
#        mask = w > 0.99
#        assert mask.sum() > 3
#        assert_allclose(np.abs(h[mask]), 0., atol=1e-4)

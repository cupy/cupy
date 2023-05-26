import pytest
from pytest import raises as assert_raises

from cupy import testing
from cupyx.scipy import signal


class TestIIRFilter:

    @pytest.mark.parametrize("N", list(range(1, 26)))
    @pytest.mark.parametrize("ftype", ['butter',
                                       pytest.param('bessel', marks=pytest.mark.xfail(
                                           reason="not implemented")),
                                       'cheby1', 'cheby2', 'ellip'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=3e-7)
    def test_symmetry(self, N, ftype, xp, scp):
        # All built-in IIR filters are real, so should have perfectly
        # symmetrical poles and zeros. Then ba representation (using
        # numpy.poly) will be purely real instead of having negligible
        # imaginary parts.
        z, p, k = scp.signal.iirfilter(N, 1.1, 1, 20, 'low', analog=True,
                                       ftype=ftype, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("N", list(range(1, 26)))
    @pytest.mark.parametrize("ftype", ['butter',
                                       pytest.param('bessel', marks=pytest.mark.xfail(
                                           reason="not implemented")),
                                       'cheby1', 'cheby2', 'ellip'])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-6, atol=5e-5)
    def test_symmetry_2(self, N, ftype, xp, scp):
        b, a = scp.signal.iirfilter(N, 1.1, 1, 20, 'low', analog=True,
                                    ftype=ftype, output='ba')
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_int_inputs(self, xp, scp):
        # Using integer frequency arguments and large N should not produce
        # numpy integers that wraparound to negative numbers
        z, p, k = scp.signal.iirfilter(24, 100, btype='low', analog=True, ftype='bessel',
                                       output='zpk')
        return z, p, k

    def test_invalid_wn_size(self):
        # low and high have 1 Wn, band and stop have 2 Wn
        assert_raises(ValueError, signal.iirfilter, 1, [0.1, 0.9], btype='low')
        assert_raises(ValueError, signal.iirfilter,
                      1, [0.2, 0.5], btype='high')
        assert_raises(ValueError, signal.iirfilter, 1, 0.2, btype='bp')
        assert_raises(ValueError, signal.iirfilter,
                      1, 400, btype='bs', analog=True)

    def test_invalid_wn_range(self):
        # For digital filters, 0 <= Wn <= 1
        assert_raises(ValueError, signal.iirfilter, 1, 2, btype='low')
        assert_raises(ValueError, signal.iirfilter, 1, [0.5, 1], btype='band')
        assert_raises(ValueError, signal.iirfilter, 1, [0., 0.5], btype='band')
        assert_raises(ValueError, signal.iirfilter, 1, -1, btype='high')
        assert_raises(ValueError, signal.iirfilter, 1, [1, 2], btype='band')
        assert_raises(ValueError, signal.iirfilter, 1, [10, 20], btype='stop')

        # analog=True with non-positive critical frequencies
        with pytest.raises(ValueError, match="must be greater than 0"):
            signal.iirfilter(2, 0, btype='low', analog=True)
        with pytest.raises(ValueError, match="must be greater than 0"):
            signal.iirfilter(2, -1, btype='low', analog=True)
        with pytest.raises(ValueError, match="must be greater than 0"):
            signal.iirfilter(2, [0, 100], analog=True)
        with pytest.raises(ValueError, match="must be greater than 0"):
            signal.iirfilter(2, [-1, 100], analog=True)
        with pytest.raises(ValueError, match="must be greater than 0"):
            signal.iirfilter(2, [10, 0], analog=True)
        with pytest.raises(ValueError, match="must be greater than 0"):
            signal.iirfilter(2, [10, -1], analog=True)

    @pytest.mark.xfail(reason="TODO: zpk2sos")
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_analog_sos(self, xp, scp):
        # first order Butterworth filter with Wn = 1 has tf 1/(s+1)
        sos = [[0., 0., 1., 0., 1., 1.]]
        sos2 = scp.signal.iirfilter(
            N=1, Wn=1, btype='low', analog=True, output='sos')
        return sos2

    def test_wn1_ge_wn0(self):
        # gh-15773: should raise error if Wn[0] >= Wn[1]
        with pytest.raises(ValueError,
                           match=r"Wn\[0\] must be less than Wn\[1\]"):
            signal.iirfilter(2, [0.5, 0.5])
        with pytest.raises(ValueError,
                           match=r"Wn\[0\] must be less than Wn\[1\]"):
            signal.iirfilter(2, [0.6, 0.5])


class TestZpk2Tf:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_identity(self, xp, scp):
        """Test the identity transfer function."""
        z = xp.array([])
        p = xp.array([])
        k = 1.
        b, a = scp.signal.zpk2tf(z, p, k)
        return b, a

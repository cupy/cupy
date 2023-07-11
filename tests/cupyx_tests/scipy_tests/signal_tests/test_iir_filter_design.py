import pytest
from pytest import raises as assert_raises

import cupy

from cupy import testing
from cupyx.scipy import signal

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


nimpl = pytest.mark.xfail(reason="not implemented")
prec_loss = pytest.mark.xfail(reason="zpk2tf loses precision")


@testing.with_requires("scipy>=1.8")
class TestIIRFilter:

    # NB: test_symmetry with higher order ellip filters need low tolerance
    # on older CUDA versions.

    @pytest.mark.parametrize("N", list(range(1, 25)))
    @pytest.mark.parametrize("ftype", ['butter',
                                       pytest.param('bessel', marks=nimpl),
                                       'cheby1', 'cheby2', 'ellip'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-6)
    def test_symmetry(self, N, ftype, xp, scp):
        # All built-in IIR filters are real, so should have perfectly
        # symmetrical poles and zeros. Then ba representation (using
        # numpy.poly) will be purely real instead of having negligible
        # imaginary parts.
        z, p, k = scp.signal.iirfilter(N, 1.1, 1, 20, 'low', analog=True,
                                       ftype=ftype, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("N", list(range(1, 25)))
    @pytest.mark.parametrize("ftype", ['butter',
                                       pytest.param('bessel', marks=nimpl),
                                       'cheby1', 'cheby2', 'ellip'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_symmetry_2(self, N, ftype, xp, scp):
        b, a = scp.signal.iirfilter(N, 1.1, 1, 20, 'low', analog=True,
                                    ftype=ftype, output='ba')
        return b, a

    @pytest.mark.xfail(reason="bessel IIR filter not implemented")
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_int_inputs(self, xp, scp):
        # Using integer frequency arguments and large N should not produce
        # numpy integers that wraparound to negative numbers
        z, p, k = scp.signal.iirfilter(24, 100, btype='low', analog=True,
                                       ftype='bessel', output='zpk')
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

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_analog_sos(self, xp, scp):
        # first order Butterworth filter with Wn = 1 has tf 1/(s+1)
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


@testing.with_requires("scipy")
class TestButter:

    @pytest.mark.parametrize('arg', [(0, 1), (1, 1)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_degenerate(self, xp, scp, arg):
        # 0-order filter is just a passthrough
        b, a = scp.signal.butter(*arg, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_degenerate_1(self, xp, scp):
        z, p, k = scp.signal.butter(1, 0.3, output='zpk')
        return z, p, k

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp, N):
        # analog s-plane
        wn = 0.01
        z, p, k = scp.signal.butter(N, wn, 'low', analog=True, output='zpk')
        return z, p, k

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic_1(self, xp, scp, N):
        # digital z-plane
        wn = 0.01
        z, p, k = scp.signal.butter(N, wn, 'high', analog=False, output='zpk')
        return z, p, k

    @pytest.mark.parametrize('arg, analog',
                             [((2, 1), True),
                              ((5, 1), True),
                                 ((10, 1), True),
                                 ((19, 1.0441379169150726), True),
                                 ((5, 0.4), False)
                              ])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic_2(self, xp, scp, arg, analog):
        b, a = scp.signal.butter(*arg, analog=analog)
        return b, a

    @pytest.mark.parametrize('arg', [(28, 0.43), (27, 0.56)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_highpass(self, xp, scp, arg):
        # highpass, high even order
        z, p, k = scp.signal.butter(*arg, 'high', output='zpk')
        return z, p, k

    @pytest.mark.parametrize("format", ['zpk', 'ba'])
    @testing.numpy_cupy_allclose(scipy_name="scp", atol=1e-12)
    def test_bandpass(self, xp, scp, format):
        output = scp.signal.butter(8, [0.25, 0.33], 'band', output=format)
        return output

    @testing.numpy_cupy_allclose(scipy_name="scp", atol=1e-12)
    def test_bandpass_analog(self, xp, scp):
        output = scp.signal.butter(
            4, [90.5, 110.5], 'bp', analog=True, output='zpk')
        return output

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_bandstop(self, xp, scp):
        z, p, k = scp.signal.butter(7, [0.45, 0.56], 'stop', output='zpk')
        z.sort()
        p.sort()
        return z, p, k

    @pytest.mark.parametrize('outp',
                             ['zpk',
                              'sos',
                              pytest.param('ba', marks=prec_loss)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    @testing.with_requires("scipy>=1.8")
    def test_ba_output(self, xp, scp, outp):
        outp = scp.signal.butter(
            4, [100, 300], 'bandpass', analog=True, output=outp)
        return outp

        # in scipy.signal, this is the output in the ba format.
        # in CUDA, zpk2tf loses precision and b,a output is garbage
        #  b2 = [1.6e+09, 0, 0, 0, 0]
        #  a2 = [1.000000000000000e+00, 5.226251859505511e+02,
        #        2.565685424949238e+05, 6.794127417357160e+07,
        #        1.519411254969542e+10, 2.038238225207147e+12,
        #        2.309116882454312e+14, 1.411088002066486e+16,
        #        8.099999999999991e+17]

    def test_fs_param(self):
        for fs in (900, 900.1, 1234.567):
            for N in (0, 1, 2, 3, 10):
                for fc in (100, 100.1, 432.12345):
                    for btype in ('lp', 'hp'):
                        ba1 = signal.butter(N, fc, btype, fs=fs)
                        ba2 = signal.butter(N, fc/(fs/2), btype)
                        testing.assert_allclose(ba1[0], ba2[0])
                        testing.assert_allclose(ba1[1], ba2[1])

                for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                    for btype in ('bp', 'bs'):
                        ba1 = signal.butter(N, fc, btype, fs=fs)
                        # for seq in (list, tuple, array):
                        fcnorm = cupy.array([f/(fs/2) for f in fc])
                        ba2 = signal.butter(N, fcnorm, btype)
                        testing.assert_allclose(ba1[0], ba2[0])
                        testing.assert_allclose(ba1[0], ba2[0])


@testing.with_requires("scipy")
class TestCheby1:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp):
        # 0-order filter is just a passthrough
        # Even-order filters have DC gain of -rp dB
        b, a = scp.signal.cheby1(0, 10*xp.log10(2), 1, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        # 1-order filter is same for all types
        b, a = scp.signal.cheby1(1, 10*xp.log10(2), 1, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_2(self, xp, scp):
        z, p, k = scp.signal.cheby1(1, 0.1, 0.3, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("N", list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        wn = 0.01
        z, p, k = scp.signal.cheby1(N, 1, wn, 'low', analog=True, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("N", list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        wn = 0.01
        z, p, k = scp.signal.cheby1(
            N, 1, wn, 'high', analog=False, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("arg, kwd",
                             [((8, 0.5, 0.048), {}),
                              ((4, 1, [0.4, 0.7]), {'btype': 'band'}),
                                 ((5, 3, 1), {'analog': True}),
                                 ((8, 0.5, 0.1), {}),
                                 ((8, 0.5, 0.25), {}),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_basic_2(self, xp, scp, arg, kwd):
        # Same test as TestNormalize
        b, a = scp.signal.cheby1(*arg, **kwd)
        return b, a

    @pytest.mark.parametrize("arg, kwd",
                             # high even order
                             [((24, 0.7, 0.2), {'output': 'zpk'}),
                              # high odd order
                              ((23, 0.8, 0.3), {'output': 'zpk'}),
                                 ((10, 1, 1000), {
                                     'analog': True, 'output': 'zpk'}),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg, kwd):
        # high even order
        z, p, k = scp.signal.cheby1(*arg, 'high', **kwd)
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        z, p, k = scp.signal.cheby1(8, 1, [0.3, 0.4], 'bp', output='zpk')
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        z, p, k = scp.signal.cheby1(7, 1, [0.5, 0.6], 'stop', output='zpk')
        z = z[xp.argsort(z.imag)]
        p = p[xp.argsort(p.imag)]
        return z, p, k

    @pytest.mark.xfail(reason='zpk2tf loses precision (cf TestButter)')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-10)
    def test_ba_output(self, xp, scp,):
        # with transfer function conversion,  without digital conversion
        b, a = scp.signal.cheby1(5, 0.9, [210, 310], 'stop', analog=True)
        return b, a

#        b2 = [1.000000000000006e+00, 0,
#              3.255000000000020e+05, 0,
#              4.238010000000026e+10, 0,
#              2.758944510000017e+15, 0,
#              8.980364380050052e+19, 0,
#              1.169243442282517e+24
#              ]
#        a2 = [1.000000000000000e+00, 4.630555945694342e+02,
#              4.039266454794788e+05, 1.338060988610237e+08,
#              5.844333551294591e+10, 1.357346371637638e+13,
#              3.804661141892782e+15, 5.670715850340080e+17,
#              1.114411200988328e+20, 8.316815934908471e+21,
#              1.169243442282517e+24
#              ]
#        assert_allclose(b, b2, rtol=1e-14)
#        assert_allclose(a, a2, rtol=1e-14)


@testing.with_requires("scipy")
class TestCheby2:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp):
        # 0-order filter is just a passthrough
        # Stopband ripple factor doesn't matter
        b, a = scp.signal.cheby2(0, 123.456, 1, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        # 1-order filter is same for all types
        b, a = scp.signal.cheby2(1, 10*xp.log10(2), 1, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_2(self, xp, scp):
        z, p, k = scp.signal.cheby2(1, 50, 0.3, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("N", list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        wn = 0.01
        z, p, k = scp.signal.cheby2(
            N, 40, wn, 'low', analog=True, output='zpk')
        return z, p, k
        #    assert_(len(p) == N)
        #    assert_(all(np.real(p) <= 0))  # No poles in right half of S-plane

    @pytest.mark.parametrize("N", list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        wn = 0.01
        z, p, k = scp.signal.cheby2(
            N, 40, wn, 'high', analog=False, output='zpk')
        return z, p, k
        #    assert_(all(np.abs(p) <= 1))  # No poles outside unit circle

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        B, A = scp.signal.cheby2(18, 100, 0.5)
        return B, A

    @pytest.mark.parametrize("arg, kwd",
                             # high even order
                             [((26, 60, 0.3), {'output': 'zpk'}),
                              # high odd order
                              ((25, 80, 0.5), {'output': 'zpk'}),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg, kwd):
        # high even order
        z, p, k = scp.signal.cheby2(*arg, 'high', **kwd)
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        z, p, k = scp.signal.cheby2(9, 40, [0.07, 0.2], 'pass', output='zpk')
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_bandstop(self, xp, scp):
        z, p, k = scp.signal.cheby2(6, 55, [0.1, 0.9], 'stop', output='zpk')
        z = z[xp.argsort(xp.angle(z))]
        p = p[xp.argsort(xp.angle(p))]
        return z, p, k

    @pytest.mark.xfail(reason='zpk2tf loses precision')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        # with transfer function conversion, without digital conversion
        b, a = scp.signal.cheby2(5, 20, [2010, 2100], 'stop', True)
        return b, a
        # b2 = [1.000000000000000e+00, 0,  # Matlab: 6.683253076978249e-12,
        #       2.111512500000000e+07, 0,  # Matlab: 1.134325604589552e-04,
        #       1.782966433781250e+14, 0,  # Matlab: 7.216787944356781e+02,
        #       7.525901316990656e+20, 0,  # Matlab: 2.039829265789886e+09,
        #       1.587960565565748e+27, 0,  # Matlab: 2.161236218626134e+15,
        #       1.339913493808585e+33]
        # a2 = [1.000000000000000e+00, 1.849550755473371e+02,
        #       2.113222918998538e+07, 3.125114149732283e+09,
        #       1.785133457155609e+14, 1.979158697776348e+16,
        #       7.535048322653831e+20, 5.567966191263037e+22,
        #       1.589246884221346e+27, 5.871210648525566e+28,
        #       1.339913493808590e+33]
        # assert_allclose(b, b2, rtol=1e-14)
        # assert_allclose(a, a2, rtol=1e-14)


@testing.with_requires("scipy>=1.8")
class TestEllip:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp):
        # 0-order filter is just a passthrough
        # Even-order filters have DC gain of -rp dB
        # Stopband ripple factor doesn't matter
        b, a = scp.signal.ellip(0, 10*xp.log10(2), 123.456, 1, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        # 1-order filter is same for all types
        b, a = scp.signal.ellip(1, 10*xp.log10(2), 1, 1, analog=True)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_2(self, xp, scp):
        z, p, k = scp.signal.ellip(1, 1, 55, 0.3, output='zpk')
        return z, p, k

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        wn = 0.01
        z, p, k = scp.signal.ellip(
            N, 1, 40, wn, 'low', analog=True, output='zpk')
        return z, p, k

    @pytest.mark.parametrize('N', list(range(20)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        wn = 0.01
        z, p, k = scp.signal.ellip(
            N, 1, 40, wn, 'high', analog=False, output='zpk')
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14, rtol=1e-14)
    def test_basic_2(self, xp, scp):
        b3, a3 = scp.signal.ellip(5, 3, 26, 1, analog=True)
        return b3, a3

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_3(self, xp, scp):
        b, a = scp.signal.ellip(3, 1, 60, [0.4, 0.7], 'stop')
        return b, a

    @pytest.mark.parametrize("arg",
                             # high even order
                             [(24, 1, 80, 0.3, 'high'),
                              # high odd order
                              (23, 1, 70, 0.5, 'high'),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg):
        # high even order
        z, p, k = scp.signal.ellip(*arg, output='zpk')
        return z, p, k

    @pytest.mark.parametrize("arg",
                             [(7, 1, 40, [0.07, 0.2], 'pass'),
                              (5, 1, 75, [90.5, 110.5], 'pass', True),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp, arg):
        z, p, k = scp.signal.ellip(7, 1, 40, [0.07, 0.2], 'pass', output='zpk')
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        z, p, k = scp.signal.ellip(8, 1, 65, [0.2, 0.4], 'stop', output='zpk')
        z = z[xp.argsort(xp.angle(z))]
        p = p[xp.argsort(xp.angle(p))]
        return z, p, k

    @pytest.mark.xfail(reason='zpk2tf loses precision')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        # with transfer function conversion,  without digital conversion
        b, a = scp.signal.ellip(5, 1, 40, [201, 240], 'stop', True)
        return b, a

#        b2 = [
#             1.000000000000000e+00, 0,  # Matlab: 1.743506051190569e-13,
#             2.426561778314366e+05, 0,  # Matlab: 3.459426536825722e-08,
#             2.348218683400168e+10, 0,  # Matlab: 2.559179747299313e-03,
#             1.132780692872241e+15, 0,  # Matlab: 8.363229375535731e+01,
#             2.724038554089566e+19, 0,  # Matlab: 1.018700994113120e+06,
#             2.612380874940186e+23
#             ]
#        a2 = [
#             1.000000000000000e+00, 1.337266601804649e+02,
#             2.486725353510667e+05, 2.628059713728125e+07,
#             2.436169536928770e+10, 1.913554568577315e+12,
#             1.175208184614438e+15, 6.115751452473410e+16,
#             2.791577695211466e+19, 7.241811142725384e+20,
#             2.612380874940182e+23
#             ]
#        assert_allclose(b, b2, rtol=1e-6)
#        assert_allclose(a, a2, rtol=1e-4)


@testing.with_requires("scipy")
class TestZpk2Tf:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_identity(self, xp, scp):
        """Test the identity transfer function."""
        z = xp.array([])
        p = xp.array([])
        k = 1.
        b, a = scp.signal.zpk2tf(z, p, k)
        return b, a

from math import sqrt, pi

import cupy
import cupyx.scipy.signal as signal
from cupyx.scipy.signal._iir_filter_conversions import _cplxreal

from cupy import testing
from cupy.testing import assert_array_almost_equal

import numpy as np

import pytest
from pytest import raises as assert_raises


@testing.with_requires("scipy")
class TestBilinear_zpk:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3

        z_d, p_d, k_d = scp.signal.bilinear_zpk(z, p, k, 10)
        return z_d, p_d, k_d

        """
        assert_allclose(sort(z_d), sort([(20-2j)/(20+2j), (20+2j)/(20-2j),
                                         -1]))
        assert_allclose(sort(p_d), sort([77/83,
                                         (1j/2 + 39/2) / (41/2 - 1j/2),
                                         (39/2 - 1j/2) / (1j/2 + 41/2), ]))
        assert_allclose(k_d, 9696/69803)
        """


@testing.with_requires("scipy")
class TestBilinear:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        b = [0.14879732743343033]
        a = [1, 0.54552236880522209, 0.14879732743343033]
        b_z, a_z = scp.signal.bilinear(b, a, 0.5)
        return b_z, a_z

#        assert_array_almost_equal(b_z, [0.087821, 0.17564, 0.087821],
#                                  decimal=5)
#        assert_array_almost_equal(a_z, [1, -1.0048, 0.35606], decimal=4)

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic_2(self, xp, scp):
        b = [1, 0, 0.17407467530697837]
        a = [1, 0.18460575326152251, 0.17407467530697837]
        b_z, a_z = scp.signal.bilinear(b, a, 0.5)
        return b_z, a_z

#        assert_array_almost_equal(b_z, [0.86413, -1.2158, 0.86413],
#                                  decimal=4)
#        assert_array_almost_equal(a_z, [1, -1.2158, 0.72826],
#                                  decimal=4)


@testing.with_requires("scipy")
class TestNormalize:

    def test_allclose(self):
        """Test for false positive on allclose in normalize() in
        filter_design.py"""
        # Test to make sure the allclose call within signal.normalize does not
        # choose false positives. Then check against a known output from MATLAB
        # to make sure the fix doesn't break anything.

        # These are the coefficients returned from
        #   `[b,a] = cheby1(8, 0.5, 0.048)'
        # in MATLAB. There are at least 15 significant figures in each
        # coefficient, so it makes sense to test for errors on the order of
        # 1e-13 (this can always be relaxed if different platforms have
        # different rounding errors)
        b_matlab = cupy.array([2.150733144728282e-11, 1.720586515782626e-10,
                               6.022052805239190e-10, 1.204410561047838e-09,
                               1.505513201309798e-09, 1.204410561047838e-09,
                               6.022052805239190e-10, 1.720586515782626e-10,
                               2.150733144728282e-11])
        a_matlab = cupy.array([1.000000000000000e+00, -7.782402035027959e+00,
                               2.654354569747454e+01, -5.182182531666387e+01,
                               6.334127355102684e+01, -4.963358186631157e+01,
                               2.434862182949389e+01, -6.836925348604676e+00,
                               8.412934944449140e-01])

        # This is the input to signal.normalize after passing through the
        # equivalent steps in signal.iirfilter as was done for MATLAB
        b_norm_in = cupy.array(
            [1.5543135865293012e-06, 1.2434508692234413e-05,
             4.3520780422820447e-05, 8.7041560845640893e-05,
             1.0880195105705122e-04, 8.7041560845640975e-05,
             4.3520780422820447e-05, 1.2434508692234413e-05,
             1.5543135865293012e-06])
        a_norm_in = cupy.array(
            [7.2269025909127173e+04, -5.6242661430467968e+05,
             1.9182761917308895e+06, -3.7451128364682454e+06,
             4.5776121393762771e+06, -3.5869706138592605e+06,
             1.7596511818472347e+06, -4.9409793515707983e+05,
             6.0799461347219651e+04])

        b_output, a_output = signal.normalize(b_norm_in, a_norm_in)

        # The test on b works for decimal=14 but the one for a does not. For
        # the sake of consistency, both of these are decimal=13. If something
        # breaks on another platform, it is probably fine to relax this lower.
        assert_array_almost_equal(b_matlab, b_output, decimal=13)
        assert_array_almost_equal(a_matlab, a_output, decimal=13)

    def test_errors(self):
        """Test the error cases."""
        # all zero denominator
        assert_raises(ValueError, signal.normalize, [1, 2], 0)

        # denominator not 1 dimensional
        assert_raises(ValueError, signal.normalize, [1, 2], [[1]])

        # numerator too many dimensions
        assert_raises(ValueError, signal.normalize, [[[1, 2]]], 1)


@testing.with_requires("scipy")
class TestLp2lp:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        b = [1]
        a = [1, float(xp.sqrt(2)), 1]
        b_lp, a_lp = scp.signal.lp2lp(b, a, 0.38574256627112119)
        return b_lp, a_lp


@testing.with_requires("scipy")
class TestLp2hp:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        b = [0.25059432325190018]
        a = [1, 0.59724041654134863, 0.92834805757524175, 0.25059432325190018]
        b_hp, a_hp = scp.signal.lp2hp(b, a, 2*pi*5000)
        return b_hp, a_hp


@testing.with_requires("scipy")
class TestLp2bp:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        b = [1]
        a = [1, 2, 2, 1]
        b_bp, a_bp = scp.signal.lp2bp(b, a, 2*pi*4000, 2*pi*2000)
        return b_bp, a_bp


@testing.with_requires("scipy")
class TestLp2bs:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        b = [1]
        a = [1, 1]
        b_bs, a_bs = scp.signal.lp2bs(
            b, a, 0.41722257286366754, 0.18460575326152251)
        return b_bs, a_bs


@testing.with_requires("scipy")
class TestLp2lp_zpk:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        z = []
        p = [(-1+1j) / sqrt(2), (-1-1j) / sqrt(2)]
        k = 1
        z_lp, p_lp, k_lp = scp.signal.lp2lp_zpk(z, p, k, 5)
        return z_lp, p_lp, k_lp

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic_2(self, xp, scp):

        # Pseudo-Chebyshev with both poles and zeros
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3
        z_lp, p_lp, k_lp = scp.signal.lp2lp_zpk(z, p, k, 20)
        return z_lp, p_lp, k_lp


@testing.with_requires("scipy")
class TestLp2hp_zpk:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        z = []
        p = [(-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2)]
        k = 1

        z_hp, p_hp, k_hp = scp.signal.lp2hp_zpk(z, p, k, 5)
        return z_hp, p_hp, k_hp

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic_2(self, xp, scp):
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3
        z_hp, p_hp, k_hp = scp.signal.lp2hp_zpk(z, p, k, 6)
        return z_hp, p_hp, k_hp


@testing.with_requires("scipy")
class TestLp2bp_zpk:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3
        z_bp, p_bp, k_bp = scp.signal.lp2bp_zpk(z, p, k, 15, 8)
        return z_bp, p_bp, k_bp


@testing.with_requires("scipy")
class TestLp2bs_zpk:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3

        z_bs, p_bs, k_bs = scp.signal.lp2bs_zpk(z, p, k, 35, 12)

        # complex sort gets confused by real parts of order +/- epsilon
        z_bs_s = z_bs[xp.argsort(z_bs.imag)]
        p_bs_s = p_bs[xp.argsort(p_bs.imag)]
        return z_bs_s, p_bs_s, k_bs


@testing.with_requires("scipy >= 1.8.0")
class TestZpk2Sos:

    @pytest.mark.parametrize('dt', 'fdFD')
    @pytest.mark.parametrize('pairing, analog',
                             [('nearest', False),
                              ('keep_odd', False),
                              ('minimal', False),
                              ('minimal', True)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dtypes(self, dt, pairing, analog, xp, scp):
        z = xp.array([-1, -1]).astype(dt)
        ct = dt.upper()  # the poles have to be complex
        p = xp.array([0.57149 + 0.29360j, 0.57149 - 0.29360j]).astype(ct)
        k = xp.array(1).astype(dt)
        sos = scp.signal.zpk2sos(z, p, k, pairing=pairing, analog=analog)
        return sos

    @pytest.mark.parametrize('case', [
        # (z, p, k)
        # cases that match octave
        ([-1, -1], [0.57149 + 0.29360j, 0.57149 - 0.29360j], 1),
        ([1j, -1j], [0.9, -0.9, 0.7j, -0.7j], 1),
        ([], [0.8, -0.5+0.25j, -0.5-0.25j], 1),
        ([1., 1., 0.9j, -0.9j],
         [0.99+0.01j, 0.99-0.01j, 0.1+0.9j, 0.1-0.9j], 1),
        ([0.9+0.1j, 0.9-0.1j, -0.9], [0.75+0.25j, 0.75-0.25j, 0.9], 1),
        # Cases that differ from octave:
        ([-0.3090 + 0.9511j, -0.3090 - 0.9511j, 0.8090 + 0.5878j,
          +0.8090 - 0.5878j, -1.0000 + 0.0000j],
         [-0.3026 + 0.9312j, -0.3026 - 0.9312j, 0.7922 + 0.5755j,
          +0.7922 - 0.5755j, -0.9791 + 0.0000j],
         1),
        ([-1 - 1.4142j, -1 + 1.4142j, -0.625 - 1.0533j, -0.625 + 1.0533j],
         [-0.2 - 0.6782j, -0.2 + 0.6782j, -0.1 - 0.5385j, -0.1 + 0.5385j],
         4),
        ([], [0.2, -0.5+0.25j, -0.5-0.25j], 1.),

    ])
    @pytest.mark.parametrize('pairing', ['nearest', 'keep_odd'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_basic(self, case, pairing, xp, scp):
        z, p, k = case
        z = xp.asarray(z)
        p = xp.asarray(p)
        sos = scp.signal.zpk2sos(z, p, k, pairing=pairing)
        return sos

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_basic_2(self, xp, scp):
        # The next two examples are adapted from Leland B. Jackson,
        # "Digital Filters and Signal Processing (1995) p.400:
        deg2rad = np.pi / 180.
        k = 1.

        # first example
        thetas = [22.5, 45, 77.5]
        mags = [0.8, 0.6, 0.9]
        z = xp.array([xp.exp(theta * deg2rad * 1j) for theta in thetas])
        z = xp.concatenate((z, z.conj()))
        p = xp.array([mag * xp.exp(theta * deg2rad * 1j)
                      for theta, mag in zip(thetas, mags)])
        p = xp.concatenate((p, p.conj()))
        sos_1 = scp.signal.zpk2sos(z, p, k)

        # second example
        z = xp.array([xp.exp(theta * deg2rad * 1j)
                      for theta in (85., 10.)])
        z = xp.concatenate((z, z.conj(), xp.array([1, -1])))
        sos_2 = scp.signal.zpk2sos(z, p, k)

        return sos_1, sos_2

    @pytest.mark.parametrize('pairing', ['nearest', 'keep_odd', 'minimal'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_pairing(self, pairing, xp, scp):
        # these examples are taken from the docstring, and show the
        # effect of the 'pairing' argument
        z1 = xp.array([-1, -0.5-0.5j, -0.5+0.5j])
        p1 = xp.array([0.75, 0.8+0.1j, 0.8-0.1j])
        sos2 = scp.signal.zpk2sos(z1, p1, 1, pairing=pairing)
        return sos2

    @pytest.mark.parametrize('p',
                             [[-1, 1, -0.1, 0.1],
                              [-0.7071+0.7071j, -0.7071-0.7071j, -0.1j, 0.1j],
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_analog(self, p, xp, scp):
        # test `analog` argument
        # for discrete time, poles closest to unit circle should appear last
        # for cont. time, poles closest to imaginary axis should appear last
        p = xp.asarray(p)
        sos2_dt = scp.signal.zpk2sos([], p, 1, pairing='minimal', analog=False)
        sos2_ct = scp.signal.zpk2sos([], p, 1, pairing='minimal', analog=True)
        return sos2_dt, sos2_ct

    def test_bad_args(self):
        with pytest.raises(ValueError, match=r'pairing must be one of'):
            signal.zpk2sos(cupy.array([1]), cupy.array(
                [2]), 1, pairing='no_such_pairing')

        with pytest.raises(ValueError, match=r'.*pairing must be "minimal"'):
            signal.zpk2sos(cupy.array([1]), cupy.array(
                [2]), 1, pairing='keep_odd', analog=True)

        with pytest.raises(ValueError,
                           match=r'.*must have len\(p\)>=len\(z\)'):
            signal.zpk2sos(cupy.array([1, 1]), cupy.array([2]), 1, analog=True)

        with pytest.raises(ValueError, match=r'k must be real'):
            signal.zpk2sos(cupy.array([1]), cupy.array([2]), k=1j)


@testing.with_requires("scipy")
class TestTf2zpk:

    @pytest.mark.parametrize('dt', ('float64', 'complex128'))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple(self, xp, scp, dt):
        z_r = xp.array([0.5, -0.5])
        p_r = xp.array([1.j / sqrt(2), -1.j / sqrt(2)])
        b = xp.poly(z_r).astype(dt)
        a = xp.poly(p_r).real.astype(dt)

        z, p, k = scp.signal.tf2zpk(b, a)
        z.sort()
        # The real part of `p` is ~0.0, so sort by imaginary part
        p = p[xp.argsort(p.imag)]
        return z, p, k


@testing.with_requires("scipy")
class TestSS2TF:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        # Test a round trip through tf2ss and ss2tf.
        b = xp.array([1.0, 3.0, 5.0])
        a = xp.array([1.0, 2.0, 3.0])

        A, B, C, D = scp.signal.tf2ss(b, a)
        return A, B, C, D

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        b = xp.array([1.0, 3.0, 5.0])
        a = xp.array([1.0, 2.0, 3.0])
        A, B, C, D = scp.signal.tf2ss(b, a)

        bb, aa = scp.signal.ss2tf(A, B, C, D)
        return bb, aa

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip(self, xp, scp):
        # See gh-5760
        tf = (2, 1)
        A, B, C, D = scp.signal.tf2ss(*tf)
        return A, B, C, D

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip_2(self, xp, scp):
        tf = (2, 1)
        A, B, C, D = scp.signal.tf2ss(*tf)

        num, den = scp.signal.ss2tf(A, B, C, D)
        return num, den

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip_3(self, xp, scp):
        tf = (xp.asarray([[5], [2]]), 1)
        A, B, C, D = scp.signal.tf2ss(*tf)
        return A, B, C, D

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip_4(self, xp, scp):
        tf = (xp.asarray([[5], [2]]), 1)
        A, B, C, D = scp.signal.tf2ss(*tf)

        num, den = scp.signal.ss2tf(A, B, C, D)
        return num, den

    @pytest.mark.parametrize('tf',
                             [([[1, 2], [1, 1]], [1, 2]),
                              ([[1, 0, 1], [1, 1, 1]], [1, 1, 1]),
                                 ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4]),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simo_round_trip(self, xp, scp, tf):
        # See gh-5753
        tf = tuple(xp.asarray(x) for x in tf)
        A, B, C, D = scp.signal.tf2ss(*tf)
        return A, B, C, D

    @pytest.mark.parametrize('tf',
                             [([[1, 2], [1, 1]], [1, 2]),
                              ([[1, 0, 1], [1, 1, 1]], [1, 1, 1]),
                                 ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4]),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_simo_round_trip_2(self, xp, scp, tf):
        tf = tuple(xp.asarray(x) for x in tf)
        A, B, C, D = scp.signal.tf2ss(*tf)

        num, den = scp.signal.ss2tf(A, B, C, D)
        return num, den

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_all_int_arrays(self, xp, scp):
        A = xp.asarray([[0, 1, 0], [0, 0, 1], [-3, -4, -2]])
        B = xp.asarray([[0], [0], [1]])
        C = xp.asarray([[5, 1, 0]])
        D = xp.asarray([[0]])
        num, den = scp.signal.ss2tf(A, B, C, D)
        return num, den

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_multioutput(self, xp, scp):
        # Regression test for gh-2669.

        # 4 states
        A = xp.array([[-1.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 2.0, 0.0],
                      [-4.0, 0.0, 3.0, 0.0],
                      [-8.0, 8.0, 0.0, 4.0]])

        # 1 input
        B = xp.array([[0.3],
                      [0.0],
                      [7.0],
                      [0.0]])

        # 3 outputs
        C = xp.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [8.0, 8.0, 0.0, 0.0]])

        D = xp.array([[0.0],
                      [0.0],
                      [1.0]])

        # Get the transfer functions for all the outputs in one call.
        b_all, a = scp.signal.ss2tf(A, B, C, D)
        return b_all, a


@testing.with_requires('scipy')
class TestSos2Zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        sos = xp.asarray([[1, 0, 1, 1, 0, -0.81],
                          [1, 0, 0, 1, 0, +0.49]])
        z, p, k = scp.signal.sos2zpk(sos)
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        sos = [[1.00000, +0.61803, 1.0000, 1.00000, +0.60515, 0.95873],
               [1.00000, -1.61803, 1.0000, 1.00000, -1.58430, 0.95873],
               [1.00000, +1.00000, 0.0000, 1.00000, +0.97915, 0.00000]]
        sos = xp.asarray(sos)
        z, p, k = scp.signal.sos2zpk(sos)
        return z, p, k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_3(self, xp, scp):
        sos = xp.array([[1, 2, 3, 1, 0.2, 0.3],
                        [4, 5, 6, 1, 0.4, 0.5]])
        z2, p2, k2 = scp.signal.sos2zpk(sos)
        return z2, p2, k2

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_fewer_zeros(self, xp, scp):
        """Test not the expected number of p/z (effectively at origin)."""
        sos = scp.signal.butter(3, 0.1, output='sos')
        z, p, k = scp.signal.sos2zpk(sos)
        return z, p, k

    def test_fewer_zeros_2(self):
        sos = signal.butter(12, [5., 30.], 'bandpass', fs=1200., analog=False,
                            output='sos')
        with pytest.warns(signal.BadCoefficients, match='Badly conditioned'):
            z, p, k = signal.sos2zpk(sos)

        assert len(z) == 24
        assert len(p) == 24


@testing.with_requires("scipy")
class TestSos2Tf:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        sos = xp.array([[1, 1, 1, 1, 0, -1],
                        [-2, 3, 1, 1, 10, 1]])
        b, a = scp.signal.sos2tf(sos)
        return b, a


@testing.with_requires("scipy")
class TestCplxReal:
    # _cplxreal is a private function, vendored from
    # scipy.signal._filter_design. This test class is also vendored.
    def test_trivial_input(self):
        assert all(x.size == 0 for x in _cplxreal([]))

        cplx1 = _cplxreal(1)
        assert cplx1[0].size == 0
        testing.assert_allclose(cplx1[1], cupy.array([1]))

    def test_output_order(self):
        # zc, zr = _cplxreal(np.roots(array([1, 0, 0, 1])))
        # assert_allclose(np.append(zc, zr), [1/2 + 1j*sin(pi/3), -1])

        eps = cupy.finfo(float).eps  # spacing(1)

        a = [0+1j, 0-1j, eps + 1j, eps - 1j, -eps + 1j, -eps - 1j,
             1, 4, 2, 3, 0, 0,
             2+3j, 2-3j,
             1-eps + 1j, 1+2j, 1-2j, 1+eps - 1j,  # sorts out of order
             3+1j, 3+1j, 3+1j, 3-1j, 3-1j, 3-1j,
             2-3j, 2+3j]
        a = cupy.array(a)
        zc, zr = _cplxreal(a)
        testing.assert_allclose(zc, [1j, 1j, 1j, 1+1j, 1+2j, 2+3j, 2+3j, 3+1j,
                                     3+1j, 3+1j])
        testing.assert_allclose(zr, [0, 0, 1, 2, 3, 4])

        z = cupy.array([1-eps + 1j, 1+2j, 1-2j, 1+eps - 1j, 1+eps+3j,
                        1-2*eps-3j, 0+1j, 0-1j, 2+4j, 2-4j, 2+3j, 2-3j, 3+7j,
                        3-7j, 4-eps+1j, 4+eps-2j, 4-1j, 4-eps+2j])

        zc, zr = _cplxreal(z)
        testing.assert_allclose(zc, [1j, 1+1j, 1+2j, 1+3j, 2+3j, 2+4j, 3+7j,
                                     4+1j, 4+2j])
        assert zr.size == 0

    def test_unmatched_conjugates(self):
        # 1+2j is unmatched
        assert_raises(ValueError, _cplxreal, [1+3j, 1-3j, 1+2j])

        # 1+2j and 1-3j are unmatched
        assert_raises(ValueError, _cplxreal, [1+3j, 1-3j, 1+2j, 1-3j])

        # 1+3j is unmatched
        assert_raises(ValueError, _cplxreal, [1+3j, 1-3j, 1+3j])

        # No pairs
        assert_raises(ValueError, _cplxreal, [1+3j])
        assert_raises(ValueError, _cplxreal, [1-3j])

    def test_real_integer_input(self):
        zc, zr = _cplxreal([2, 0, 1, 4])
        assert zc.size == 0
        testing.assert_allclose(zr, [0, 1, 2, 4], atol=1e-15)


@testing.with_requires("scipy")
class TestLowLevelAP:
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_buttap(self, xp, scp):
        return scp.signal.buttap(3)

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_cheb1ap(self, xp, scp):
        return scp.signal.cheb1ap(3, 1)

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_cheb2ap(self, xp, scp):
        return scp.signal.cheb2ap(3, 1)

    @testing.numpy_cupy_allclose(scipy_name="scp", atol=2e-4, rtol=2e-4)
    def test_ellipap(self, xp, scp):
        return scp.signal.ellipap(7, 1, 10)

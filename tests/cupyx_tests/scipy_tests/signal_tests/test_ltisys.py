import platform
import sys

import cupy
from cupyx.scipy.signal import abcd_normalize
import cupyx.scipy.signal as signal

import pytest
from pytest import raises as assert_raises
from cupy import testing

try:
    import scipy.signal   # NOQA
except ImportError:
    pass


def assert_equal(actual, desired):
    try:
        assert bool((actual == desired).all())
    except AttributeError:
        assert actual == desired


class Test_abcd_normalize:
    def setup_method(self):
        self.A = cupy.array([[1.0, 2.0], [3.0, 4.0]])
        self.B = cupy.array([[-1.0], [5.0]])
        self.C = cupy.array([[4.0, 5.0]])
        self.D = cupy.array([[2.5]])

    def test_no_matrix_fails(self):
        assert_raises(ValueError, abcd_normalize)

    def test_A_nosquare_fails(self):
        assert_raises(ValueError, abcd_normalize, [1, -1],
                      self.B, self.C, self.D)

    def test_AB_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_AC_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      [[4.0], [5.0]], self.D)

    def test_CD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      self.C, [2.5, 0])

    def test_BD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_normalized_matrices_unchanged(self):
        A, B, C, D = abcd_normalize(self.A, self.B, self.C, self.D)
        assert_equal(A, self.A)
        assert_equal(B, self.B)
        assert_equal(C, self.C)
        assert_equal(D, self.D)

    def test_shapes(self):
        A, B, C, D = abcd_normalize(self.A, self.B, [1, 0], 0)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape[0], C.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(B.shape[1], D.shape[1])

    def test_zero_dimension_is_not_none1(self):
        B_ = cupy.zeros((2, 0))
        D_ = cupy.zeros((0, 0))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, D=D_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(D, D_)
        assert_equal(C.shape[0], D_.shape[0])
        assert_equal(C.shape[1], self.A.shape[0])

    def test_zero_dimension_is_not_none2(self):
        B_ = cupy.zeros((2, 0))
        C_ = cupy.zeros((0, 2))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, C=C_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(C, C_)
        assert_equal(D.shape[0], C_.shape[0])
        assert_equal(D.shape[1], B_.shape[1])

    def test_missing_A(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))

    def test_missing_B(self):
        A, B, C, D = abcd_normalize(A=self.A, C=self.C, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))

    def test_missing_C(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, D=self.D)
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_D(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, C=self.C)
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_AB(self):
        A, B, C, D = abcd_normalize(C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(A.shape, (self.C.shape[1], self.C.shape[1]))
        assert_equal(B.shape, (self.C.shape[1], self.D.shape[1]))

    def test_missing_AC(self):
        A, B, C, D = abcd_normalize(B=self.B, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(C.shape, (self.D.shape[0], self.B.shape[0]))

    def test_missing_AD(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_BC(self):
        A, B, C, D = abcd_normalize(A=self.A, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_ABC_fails(self):
        assert_raises(ValueError, abcd_normalize, D=self.D)

    def test_missing_BD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, C=self.C)

    def test_missing_CD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, B=self.B)


@testing.with_requires("scipy")
class Test_bode:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_01(self, xp, scp):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        # cutoff: 1 rad/s, slope: -20 dB/decade
        #   H(s=0.1) ~= 0 dB
        #   H(s=1) ~= -3 dB
        #   H(s=10) ~= -20 dB
        #   H(s=100) ~= -40 dB
        system = scp.signal.lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = scp.signal.bode(system, w=w)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_02(self, xp, scp):
        # Test bode() phase calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   angle(H(s=0.1)) ~= -5.7 deg
        #   angle(H(s=1)) ~= -45 deg
        #   angle(H(s=10)) ~= -84.3 deg
        system = scp.signal.lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, mag, phase = scp.signal.bode(system, w=w)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_03(self, xp, scp):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = scp.signal.lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = scp.signal.bode(system, w=w)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_04(self, xp, scp):
        # Test bode() phase calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = scp.signal.lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = scp.signal.bode(system, w=w)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_05(self, xp, scp):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = scp.signal.lti([1], [1, 1])
        n = 10
        w, mag, phase = scp.signal.bode(system, n=n)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_06(self, xp, scp):
        # Test that bode() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = scp.signal.lti([1], [1, 0])
        w, mag, phase = scp.signal.bode(system, n=2)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-10)
    def test_07(self, xp, scp):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        system = scp.signal.lti([1], [1, 0, 100])
        w, mag, phase = scp.signal.bode(system, n=2)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_08(self, xp, scp):
        # Test that bode() return continuous phase, issues/2331.
        system = scp.signal.lti([], [-10, -30, -40, -60, -70], 1)
        w, mag, phase = system.bode(w=xp.logspace(-3, 40, 100))
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_from_state_space(self, xp, scp):
        # Ensure that bode works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n)
        # is the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        a = xp.array([1.0, 2.0, 2.0, 1.0])
        A = scp.linalg.companion(a).T
        B = xp.array([[0.0], [0.0], [1.0]])
        C = xp.array([[1.0, 0.0, 0.0]])
        D = xp.array([[0.0]])
        system = scp.signal.lti(A, B, C, D)
        w, mag, phase = scp.signal.bode(system, n=100)
        return w, mag, phase


@testing.with_requires("scipy")
class Test_freqresp:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_output_manual(self, xp, scp):
        # Test freqresp() output calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   re(H(s=0.1)) ~= 0.99
        #   re(H(s=1)) ~= 0.5
        #   re(H(s=10)) ~= 0.0099
        system = scp.signal.lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, H = scp.signal.freqresp(system, w=w)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_output(self, xp, scp):
        # Test freqresp() output calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = scp.signal.lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, H = scp.signal.freqresp(system, w=w)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_freq_range(self, xp, scp):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Expected range is from 0.01 to 10.
        system = scp.signal.lti([1], [1, 1])
        n = 10
        w, H = scp.signal.freqresp(system, n=n)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pole_zero(self, xp, scp):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = scp.signal.lti([1], [1, 0])
        w, H = scp.signal.freqresp(system, n=2)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_from_state_space(self, xp, scp):
        # Ensure that freqresp works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n) is
        # the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        a = xp.array([1.0, 2.0, 2.0, 1.0])
        A = scp.linalg.companion(a).T
        B = xp.array([[0.0], [0.0], [1.0]])
        C = xp.array([[1.0, 0.0, 0.0]])
        D = xp.array([[0.0]])
        system = scp.signal.lti(A, B, C, D)
        w, H = scp.signal.freqresp(system, n=10)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_from_zpk(self, xp, scp):
        # 4th order low-pass filter: H(s) = 1 / (s + 1)
        system = scp.signal.lti([], [-1]*4, [1])
        w = [0.1, 1, 10, 100]
        w, H = scp.signal.freqresp(system, w=w)
        return w, H


@testing.with_requires("scipy")
class TestLsim:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order(self, xp, scp):
        # y' = -y
        # exact solution is y(t) = exp(-t)
        # system = self.lti_nowarn(-1.,1.,1.,0.)
        system = scp.signal.lti(-1., 1., 1., 0.)
        t = xp.linspace(0, 5)
        u = xp.zeros_like(t)
        tout, y, x = scp.signal.lsim(system, u, t, X0=xp.asarray([1.0]))
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_second_order(self, xp, scp):
        t = xp.linspace(0, 10, 1001)
        u = xp.zeros_like(t)
        # Second order system with a repeated root: x''(t) + 2*x(t) + x(t) = 0.
        # With initial conditions x(0)=1.0 and x'(t)=0.0, the exact solution
        # is (1-t)*exp(-t).
        system = scp.signal.lti([1.0], [1.0, 2.0, 1.0])
        tout, y, x = scp.signal.lsim(system, u, t, X0=xp.asarray([1.0, 0.0]))
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrator(self, xp, scp):
        # integrator: y' = u
        system = scp.signal.lti(0., 1., 1., 0.)
        t = xp.linspace(0, 5)
        u = t
        tout, y, x = scp.signal.lsim(system, u, t)
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_two_states(self, xp, scp):
        # A system with two state variables, two inputs, and one output.
        A = xp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = xp.array([[1.0, 0.0], [0.0, 1.0]])
        C = xp.array([1.0, 0.0])
        D = xp.zeros((1, 2))

        system = scp.signal.lti(A, B, C, D)

        t = xp.linspace(0, 10.0, 21)
        u = xp.zeros((len(t), 2))
        tout, y, x = scp.signal.lsim(
            system, U=u, T=t, X0=xp.asarray([1.0, 1.0]))
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_double_integrator(self, xp, scp):
        # double integrator: y'' = 2u
        A = xp.array([[0., 1.], [0., 0.]])
        B = xp.array([[0.], [1.]])
        C = xp.array([[2., 0.]])
        system = scp.signal.lti(A, B, C, 0.)
        t = xp.linspace(0, 5)
        u = xp.ones_like(t)
        tout, y, x = scp.signal.lsim(system, u, t)
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_jordan_block(self, xp, scp):
        # Non-diagonalizable A matrix
        #   x1' + x1 = x2
        #   x2' + x2 = u
        #   y = x1
        # Exact solution with u = 0 is y(t) = t exp(-t)
        A = xp.array([[-1., 1.], [0., -1.]])
        B = xp.array([[0.], [1.]])
        C = xp.array([[1., 0.]])
        system = scp.signal.lti(A, B, C, 0.)
        t = xp.linspace(0, 5)
        u = xp.zeros_like(t)
        tout, y, x = scp.signal.lsim(system, u, t, X0=xp.asarray([0.0, 1.0]))
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_miso(self, xp, scp):
        # A system with two state variables, two inputs, and one output.
        A = xp.array([[-1.0, 0.0], [0.0, -2.0]])
        B = xp.array([[1.0, 0.0], [0.0, 1.0]])
        C = xp.array([1.0, 0.0])
        D = xp.zeros((1, 2))
        system = scp.signal.lti(A, B, C, D)

        t = xp.linspace(0, 5.0, 101)
        u = xp.zeros((len(t), 2))
        tout, y, x = scp.signal.lsim(system, u, t, X0=xp.asarray([1.0, 1.0]))
        return tout, y, x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nonzero_initial_time(self, xp, scp):
        system = scp.signal.lti(-1., 1., 1., 0.)
        t = xp.linspace(1, 2)
        u = xp.zeros_like(t)
        tout, y, x = scp.signal.lsim(system, u, t, X0=xp.array([1.0]))
        return tout, y, x

    def test_nonequal_timesteps(self):
        t = cupy.array([0.0, 1.0, 1.0, 3.0])
        u = cupy.array([0.0, 0.0, 1.0, 1.0])
        # Simple integrator: x'(t) = u(t)
        system = ([1.0], [1.0, 0.0])
        with assert_raises(ValueError,
                           match="Time steps are not equally spaced."):
            signal.lsim(system, u, t, X0=cupy.array([1.0]))


@testing.with_requires("scipy")
class TestImpulse:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order(self, xp, scp):
        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        system = ([1.0], [1.0, 1.0])
        tout, y = scp.signal.impulse(system)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order_fixed_time(self, xp, scp):
        # Specify the desired time values for the output.

        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        system = ([1.0], [1.0, 1.0])
        n = 21
        t = xp.linspace(0, 2.0, n)
        tout, y = scp.signal.impulse(system, T=t)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order_initial(self, xp, scp):
        # Specify an initial condition as a scalar.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        system = ([1.0], [1.0, 1.0])
        tout, y = scp.signal.impulse(system, X0=3.0)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order_initial_list(self, xp, scp):
        # Specify an initial condition as a list.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        system = ([1.0], [1.0, 1.0])
        tout, y = scp.signal.impulse(system, X0=xp.asarray([3.0]))
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrator(self, xp, scp):
        # Simple integrator: x'(t) = u(t)
        system = ([1.0], [1.0, 0.0])
        tout, y = scp.signal.impulse(system)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_second_order(self, xp, scp):
        # Second order system with a repeated root:
        #     x''(t) + 2*x(t) + x(t) = u(t)
        # The exact impulse response is t*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = scp.signal.impulse(system)
        return tout, y


@testing.with_requires("scipy")
class TestStep:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order(self, xp, scp):
        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        system = ([1.0], [1.0, 1.0])
        tout, y = scp.signal.step(system)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order_fixed_time(self, xp, scp):
        # Specify the desired time values for the output.

        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        system = ([1.0], [1.0, 1.0])
        n = 21
        t = xp.linspace(0, 2.0, n)
        tout, y = scp.signal.step(system, T=t)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order_initial(self, xp, scp):
        # Specify an initial condition as a scalar.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0, 1.0])
        tout, y = scp.signal.step(system, X0=3.0)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_first_order_initial_list(self, xp, scp):
        # Specify an initial condition as a list.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0, 1.0])
        tout, y = scp.signal.step(system, X0=xp.array([3.0]))
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrator(self, xp, scp):
        # Simple integrator: x'(t) = u(t)
        # Exact step response is x(t) = t.
        system = ([1.0], [1.0, 0.0])
        tout, y = scp.signal.step(system)
        return tout, y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_second_order(self, xp, scp):
        # Second order system with a repeated root:
        #     x''(t) + 2*x(t) + x(t) = u(t)
        # The exact step response is 1 - (1 + t)*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = scp.signal.step(system)
        return tout, y


@testing.with_requires('scipy')
class TestPlacePoles:

    @pytest.mark.parametrize('method', ['KNV0', 'YT'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_real(self, xp, scp, method):
        # Test real pole placement using KNV and YT0 algorithm and example 1 in
        # section 4 of the reference publication (see place_poles docstring)
        A = xp.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = xp.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0]).reshape(4, 2)
        P = xp.array([-0.2, -0.5, -5.0566, -8.6659])

        fsf = scp.signal.place_poles(A, B, P, method=method)
        return fsf.computed_poles

        # Check that both KNV and YT compute correct K matrix
        self._check(A, B, P, method='KNV0')
        self._check(A, B, P, method='YT')

    @pytest.mark.xfail(
        sys.platform.startswith('win32')
        or platform.processor() == "aarch64",
        reason='passes locally, fails on windows CI, aarch64')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_real_2(self, xp, scp):
        # Try to reach the specific case in _YT_real where two singular
        # values are almost equal. This is to improve code coverage but I
        # have no way to be sure this code is really reached

        A = xp.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = xp.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0]).reshape(4, 2)

        fsf = scp.signal.place_poles(A, B, xp.asarray([2, 2, 3, 3]))
        poles = xp.real_if_close(fsf.computed_poles)
        p = poles.copy()    # make contiguous
        p.sort()
        return p

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex(self, xp, scp):
        # Test complex pole placement on a linearized car model, taken from L.
        # Jaulin, Automatique pour la robotique, Cours et Exercices, iSTE
        # editions p 184/185
        A = xp.array([[0, 7, 0, 0],
                      [0, 0, 0, 7/3.],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        B = xp.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        # Test complex poles on YT
        P = xp.array([-3, -1, -2-1j, -2+1j])

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    @testing.with_requires("scipy >= 1.9")
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-8)
    def test_complex_2(self, xp, scp):
        # Try to reach the specific case in _YT_complex where two singular
        # values are almost equal. This is to improve code coverage but I
        # have no way to be sure this code is really reached
        A = xp.array([[0, 7, 0, 0],
                      [0, 0, 0, 7/3.],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        B = xp.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        P = xp.array([0-1e-6j, 0+1e-6j, -10, 10])

        fsf = scp.signal.place_poles(A, B, P, maxiter=1000)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_3(self, xp, scp):
        # Try to reach the specific case in _YT_complex where the rank two
        # update yields two null vectors. This test was found via Monte Carlo.

        A = xp.array(
            [-2148, -2902, -2267, -598, -1722, -1829, -165, -283, -2546,
             -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
             -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
             -291, -338, -153, -1804, -1106, -1168, -867, -2297]
        ).reshape(6, 6)

        B = xp.array(
            [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
             -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
             -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
             -491, -1543, -686]
        ).reshape(6, 5)
        P = xp.array([-25.-29.j, -25.+29.j, 31.-42.j,
                     31.+42.j, 33.-41.j, 33.+41.j])

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    @pytest.mark.skip(reason="numerical stability: scipy QR vs numpy QR")
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_4(self, xp, scp):
        # Use a lot of poles to go through all cases for update_order
        # in _YT_loop

        A = xp.array(
            [-2148, -2902, -2267, -598, -1722, -1829, -165, -283, -2546,
             -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
             -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
             -291, -338, -153, -1804, -1106, -1168, -867, -2297]
        ).reshape(6, 6)

        B = xp.array(
            [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
             -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
             -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
             -491, -1543, -686]
        ).reshape(6, 5)

        big_A = xp.ones((11, 11)) - xp.eye(11)
        big_B = xp.ones((11, 10)) - xp.diag([1]*10, 1)[:, 1:]
        big_A[:6, :6] = A
        big_B[:6, :5] = B

        P = xp.array([-10, -20, -30, 40, 50, 60, 70, -20 - 5j, -20 + 5j,
                      5 + 3j, 5 - 3j])
        fsf = scp.signal.place_poles(big_A, big_B, P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_5(self, xp, scp):
        big_A = xp.ones((11, 11)) - xp.eye(11)
        big_B = xp.ones((11, 10)) - xp.diag([1]*10, 1)[:, 1:]

        A = xp.array(
            [-2148, -2902, -2267, -598, -1722, -1829, -165, -283, -2546,
             -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
             -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
             -291, -338, -153, -1804, -1106, -1168, -867, -2297]
        ).reshape(6, 6)

        B = xp.array(
            [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
             -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
             -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
             -491, -1543, -686]
        ).reshape(6, 5)

        big_A[:6, :6] = A
        big_B[:6, :5] = B

        # check with only complex poles and only real poles
        P = xp.asarray([-10, -20, -30, -40, -50, -60, -70, -80, -90, -100])

        fsf = scp.signal.place_poles(big_A[:-1, :-1], big_B[:-1, :-1], P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_6(self, xp, scp):
        big_A = xp.ones((11, 11)) - xp.eye(11)
        big_B = xp.ones((11, 10)) - xp.diag([1]*10, 1)[:, 1:]

        A = xp.array(
            [-2148, -2902, -2267, -598, -1722, -1829, -165, -283, -2546,
             -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
             -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
             -291, -338, -153, -1804, -1106, -1168, -867, -2297]
        ).reshape(6, 6)

        B = xp.array(
            [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
             -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
             -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
             -491, -1543, -686]
        ).reshape(6, 5)

        big_A[:6, :6] = A
        big_B[:6, :5] = B

        P = xp.asarray([-10, -20, -30, -40, -50, -60, -70, -80, -90, -100])

        fsf = scp.signal.place_poles(big_A[:-1, :-1], big_B[:-1, :-1], P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_7(self, xp, scp):
        big_A = xp.ones((11, 11)) - xp.eye(11)
        big_B = xp.ones((11, 10)) - xp.diag([1]*10, 1)[:, 1:]

        A = xp.array(
            [-2148, -2902, -2267, -598, -1722, -1829, -165, -283, -2546,
             -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
             -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
             -291, -338, -153, -1804, -1106, -1168, -867, -2297]
        ).reshape(6, 6)

        B = xp.array(
            [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
             -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
             -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
             -491, -1543, -686]
        ).reshape(6, 5)

        big_A[:6, :6] = A
        big_B[:6, :5] = B

        P = xp.asarray([-10+10j, -20+20j, -30+30j, -40+40j, -50+50j,
                        -10-10j, -20-20j, -30-30j, -40-40j, -50-50j])

        fsf = scp.signal.place_poles(big_A[:-1, :-1], big_B[:-1, :-1], P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_8(self, xp, scp):
        # need a 5x5 array to ensure YT handles properly when there
        # is only one real pole and several complex
        A = xp.array([0, 7, 0, 0, 0, 0, 0, 7/3., 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 5, 0, 0, 0, 0, 9]).reshape(5, 5)
        B = xp.array([0, 0, 0, 0, 1, 0, 0, 1, 2, 3]).reshape(5, 2)
        P = xp.array([-2, -3+1j, -3-1j, -1+1j, -1-1j])

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_9(self, xp, scp):
        # need a 5x5 array to ensure YT handles properly when there
        # is only one real pole and several complex
        A = xp.array([0, 7, 0, 0, 0, 0, 0, 7/3., 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 5, 0, 0, 0, 0, 9]).reshape(5, 5)
        B = xp.array([0, 0, 0, 0, 1, 0, 0, 1, 2, 3]).reshape(5, 2)

        # same test with an odd number of real poles > 1
        # this is another specific case of YT
        P = xp.array([-2, -3, -4, -1+1j, -1-1j])

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tricky_B(self, xp, scp):
        # check we handle as we should the 1 column B matrices and
        # n column B matrices (with n such as shape(A)=(n, n))
        A = xp.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = xp.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4,
                      5, 6, 7, 8]).reshape(4, 4)

        # KNV or YT are not called here, it's a specific case with only
        # one unique solution
        P = xp.array([-0.2, -0.5, -5.0566, -8.6659])

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tricky_B_2(self, xp, scp):
        # check we handle as we should the 1 column B matrices and
        # n column B matrices (with n such as shape(A)=(n, n))
        A = xp.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = xp.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4,
                      5, 6, 7, 8]).reshape(4, 4)

        # check with complex poles too as they trigger a specific case in
        # the specific case :-)
        P = xp.array((-2+1j, -2-1j, -3, -2))

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tricky_B_3(self, xp, scp):
        # check we handle as we should the 1 column B matrices and
        # n column B matrices (with n such as shape(A)=(n, n))
        A = xp.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = xp.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4,
                      5, 6, 7, 8]).reshape(4, 4)

        # now test with a B matrix with only one column (no optimisation)
        B = B[:, 0].reshape(4, 1)
        P = xp.array((-2+1j, -2-1j, -3, -2))

        fsf = scp.signal.place_poles(A, B, P)
        return fsf.computed_poles

    def test_errors(self):
        # Test input mistakes from user
        A = cupy.array([0, 7, 0, 0, 0, 0, 0, 7/3., 0, 0,
                       0, 0, 0, 0, 0, 0]).reshape(4, 4)
        B = cupy.array([0, 0, 0, 0, 1, 0, 0, 1]).reshape(4, 2)

        # should fail as the method keyword is invalid
        with assert_raises(ValueError):
            signal.place_poles(A, B, cupy.array([-2.1, -2.2, -2.3, -2.4]),
                               method="foo")

        # should fail as poles are not 1D array
        assert_raises(ValueError, signal.place_poles, A, B,
                      cupy.array((-2.1, -2.2, -2.3, -2.4)).reshape(4, 1))

        # should fail as A is not a 2D array
        assert_raises(ValueError, signal.place_poles, A[:, :, None], B,
                      cupy.array([-2.1, -2.2, -2.3, -2.4]))

        # should fail as B is not a 2D array
        assert_raises(ValueError, signal.place_poles, A, B[:, :, None],
                      cupy.array([-2.1, -2.2, -2.3, -2.4]))

        # should fail as there are too many poles
        assert_raises(ValueError, signal.place_poles, A, B,
                      cupy.array([-2.1, -2.2, -2.3, -2.4, -3]))

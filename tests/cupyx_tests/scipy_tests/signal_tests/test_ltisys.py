import cupy
from cupyx.scipy.signal import abcd_normalize
import cupyx.scipy.signal as signal

from cupy import testing
from pytest import raises as assert_raises


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

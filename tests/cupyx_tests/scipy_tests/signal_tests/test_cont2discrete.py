import math

import cupyx.scipy.signal as signal  # noqa

import pytest
from cupy import testing


@testing.with_requires("scipy")
class TestC2D:

    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_zoh(self, xp, scp):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [-0.33]])
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='zoh')
        return ad, bd, cd, dd, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_foh(self, xp, scp):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [-0.33]])
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='foh')
        return ad, bd, cd, dd, dt

    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_impulse(self, xp, scp):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [0.0]])
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='impulse')
        return ad, bd, cd, dd, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_gbt(self, xp, scp):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [-0.33]])

        dt_requested = 0.5
        alpha = 1.0 / 3.0

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='gbt', alpha=alpha)
        return ad, bd, cd, dd, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_euler(self, xp, scp):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [-0.33]])
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='euler')
        return ad, bd, cd, dd, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_backward_diff(self, xp, scp):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [-0.33]])
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='backward_diff')
        return ad, bd, cd, dd, dt

    @pytest.mark.parametrize('dt_requested', [0.5, 1.0 / 3.0])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bilinear(self, xp, scp, dt_requested):
        ac = xp.eye(2)
        bc = xp.full((2, 1), 0.5)
        cc = xp.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = xp.array([[0.0], [0.0], [-0.33]])

        c2d = scp.signal.cont2discrete
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='bilinear')
        return ad, bd, cd, dd, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_transferfunction(self, xp, scp):
        numc = xp.array([0.25, 0.25, 0.5])
        denc = xp.array([0.75, 0.75, 1.0])
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        num, den, dt = c2d((numc, denc), dt_requested, method='zoh')
        return num, den, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zerospolesgain(self, xp, scp):
        zeros_c = xp.array([0.5, -0.5])
        poles_c = xp.array([1.j / math.sqrt(2), -1.j / math.sqrt(2)])
        k_c = 1.0
        dt_requested = 0.5

        c2d = scp.signal.cont2discrete
        zeros, poles, k, dt = c2d((zeros_c, poles_c, k_c), dt_requested,
                                  method='zoh')
        return zeros, poles, k, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_gbt_with_sio_tf_and_zpk(self, xp, scp):
        """Test method='gbt' with alpha=0.25 for tf and zpk cases."""
        # State space coefficients for the continuous SIO system.
        A = -1.0
        B = 1.0
        C = 1.0
        D = 0.5

        # The continuous transfer function coefficients.
        cnum, cden = scp.signal.ss2tf(A, B, C, D)

        # Continuous zpk representation
        cz, cp, ck = scp.signal.ss2zpk(A, B, C, D)

        h = 1.0
        alpha = 0.25

        # Compute the discrete tf using cont2discrete.
        c2d = scp.signal.cont2discrete
        c2dnum, c2dden, dt = c2d((cnum, cden), h, method='gbt', alpha=alpha)
        return c2dnum, c2dden, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_discrete_approx(self, xp, scp):
        """
        Test that the solution to the discrete approximation of a continuous
        system actually approximates the solution to the continuous system.
        This is an indirect test of the correctness of the implementation
        of cont2discrete.
        """

        def u(t):
            return xp.sin(2.5 * t)

        a = xp.array([[-0.01]])
        b = xp.array([[1.0]])
        c = xp.array([[1.0]])
        d = xp.array([[0.2]])
        x0 = 1.0

        t = xp.linspace(0, 10.0, 101)
        dt = t[1] - t[0]
        u1 = u(t)

        # Use lsim to compute the solution to the continuous system.
        t, yout, xout = scp.signal.lsim((a, b, c, d), T=t, U=u1, X0=x0)

        # Convert the continuous system to a discrete approximation.
        dsys = scp.signal.cont2discrete((a, b, c, d), dt, method='bilinear')

        # Use dlsim with the pairwise averaged input to compute the output
        # of the discrete system.
        u2 = 0.5 * (u1[:-1] + u1[1:])
        t2 = t[:-1]
        td2, yd2, xd2 = scp.signal.dlsim(
            dsys, u=u2.reshape(-1, 1), t=t2, x0=x0)

        # ymid is the average of consecutive terms of the "exact" output
        # computed by lsim2.  This is what the discrete approximation
        # actually approximates.
        ymid = 0.5 * (yout[:-1] + yout[1:])

        return yd2, ymid

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simo_tf(self, xp, scp):
        # See gh-5753
        tf = ([[1, 0], [1, 1]], [1, 1])
        num, den, dt = scp.signal.cont2discrete(tf, 0.01)
        return num, den, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multioutput(self, xp, scp):
        ts = 0.01  # time step
        tf = ([[1, -3], [1, 5]], [1, 1])
        num, den, dt = scp.signal.cont2discrete(tf, ts)
        return num, den, dt

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multioutput_1(self, xp, scp):
        ts = 0.01  # time step
        tf = ([[1, -3], [1, 5]], [1, 1])

        tf1 = (tf[0][0], tf[1])
        num1, den1, dt1 = scp.signal.cont2discrete(tf1, ts)
        return num1, den1, dt1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multioutput_2(self, xp, scp):
        ts = 0.01  # time step
        tf = ([[1, -3], [1, 5]], [1, 1])

        tf2 = (tf[0][1], tf[1])
        num2, den2, dt2 = scp.signal.cont2discrete(tf2, ts)
        return num2, den2, dt2


@testing.with_requires('scipy')
class TestC2dLti:

    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_c2d_ss(self, xp, scp):
        # StateSpace
        A = xp.array([[-0.3, 0.1], [0.2, -0.7]])
        B = xp.array([[0], [1]])
        C = xp.array([[1, 0]])
        D = 0

        sys_ssc = scp.signal.lti(A, B, C, D)
        sys_ssd = sys_ssc.to_discrete(0.05)
        return sys_ssd.A, sys_ssd.B, sys_ssd.C, sys_ssd.D

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_c2d_tf(self, xp, scp):

        sys = scp.signal.lti([0.5, 0.3], [1.0, 0.4])
        sys = sys.to_discrete(0.005)
        return sys.num, sys.den


# Some test cases for checking the invariances.
# Array of triplets: (system, sample time, number of samples)
# here 'system' is a tuple to be wrapped into tfss(*system)
cases = [
    (([1, 1], [1, 1.5, 1]), 0.25, 10),
    (([1, 2], [1, 1.5, 3, 1]), 0.5, 10),
    ((0.1, [1, 1, 2, 1]), 0.5, 10),
]


@testing.with_requires('scipy')
class TestC2dInvariants:

    # Check that systems discretized with the impulse-invariant
    # method really hold the invariant
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_impulse_invariant(self, xp, scp,
                               sys, sample_time, samples_number):
        time = xp.arange(samples_number) * sample_time
        sys = scp.signal.tf2ss(*sys)

        c2d = scp.signal.cont2discrete
        _, yout_cont = scp.signal.impulse(sys, T=time)
        _, yout_disc = scp.signal.dimpulse(
            c2d(sys, sample_time, method='impulse'), n=len(time))
        return yout_cont, yout_disc[0]

    # Step invariant should hold for ZOH discretized systems
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_step_invariant(self, xp, scp, sys, sample_time, samples_number):
        time = xp.arange(samples_number) * sample_time
        sys = scp.signal.tf2ss(*sys)

        c2d = scp.signal.cont2discrete
        _, yout_cont = scp.signal.step(sys, T=time)
        _, yout_disc = scp.signal.dstep(
            c2d(sys, sample_time, method='zoh'), n=len(time))
        return yout_cont, yout_disc[0]

    # Linear invariant should hold for FOH discretized systems
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_linear_invariant(self, xp, scp, sys, sample_time, samples_number):
        time = xp.arange(samples_number) * sample_time
        sys = scp.signal.tf2ss(*sys)

        c2d = scp.signal.cont2discrete
        _, yout_cont, _ = scp.signal.lsim(sys, T=time, U=time)
        _, yout_disc, _ = scp.signal.dlsim(
            c2d(sys, sample_time, method='foh'), u=time)
        return yout_cont, yout_disc

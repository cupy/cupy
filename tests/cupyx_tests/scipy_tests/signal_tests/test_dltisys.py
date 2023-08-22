from math import sqrt, pi

from pytest import raises as assert_raises

import cupy
from cupy import testing

import cupyx.scipy.signal as signal


@testing.with_requires('scipy')
class TestDLTI:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim(self, xp, scp):
        a = xp.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = xp.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = xp.asarray([[0.1, 0.3]])
        d = xp.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        # Create an input matrix with inputs down the columns (3 cols) and its
        # respective time input vector
        u = xp.hstack((xp.linspace(0, 4.0, num=5)[:, None],
                       xp.full((5, 1), 0.01),
                       xp.full((5, 1), -0.002)))
        t_in = xp.linspace(0, 2.0, num=5)

        tout, yout, xout = scp.signal.dlsim((a, b, c, d, dt), u, t_in)
        return tout, yout, xout

    def test_dlsim_2(self):
        # Make sure input with single-dimension doesn't raise error
        signal.dlsim((1, 2, 3), 4)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_3(self, xp, scp):
        a = xp.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = xp.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = xp.asarray([[0.1, 0.3]])
        d = xp.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        # Create an input matrix with inputs down the columns (3 cols) and its
        # respective time input vector
        u = xp.hstack((xp.linspace(0, 4.0, num=5)[:, None],
                       xp.full((5, 1), 0.01),
                       xp.full((5, 1), -0.002)))

        # Interpolated control - inputs should have different time steps
        # than the discrete model uses internally
        u_sparse = u[[0, 4], :]
        t_sparse = xp.asarray([0.0, 2.0])

        tout, yout, xout = scp.signal.dlsim(
            (a, b, c, d, dt), u_sparse, t_sparse)
        return tout, yout, xout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_4(self, xp, scp):

        # Create an input matrix with inputs down the columns (3 cols) and its
        # respective time input vector
        u = xp.hstack((xp.linspace(0, 4.0, num=5)[:, None],
                       xp.full((5, 1), 0.01),
                       xp.full((5, 1), -0.002)))
        t_in = xp.linspace(0, 2.0, num=5)

        # Transfer functions (assume dt = 0.5)
        num = xp.asarray([1.0, -0.1])
        den = xp.asarray([0.3, 1.0, 0.2])

        # Assume use of the first column of the control input built earlier
        tout, yout = scp.signal.dlsim((num, den, 0.5), u[:, 0], t_in)
        return tout, yout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_5(self, xp, scp):
        # Retest the same with a 1-D input vector
        u = xp.hstack((xp.linspace(0, 4.0, num=5)[:, None],
                       xp.full((5, 1), 0.01),
                       xp.full((5, 1), -0.002)))
        t_in = xp.linspace(0, 2.0, num=5)

        # Transfer functions (assume dt = 0.5)
        num = xp.asarray([1.0, -0.1])
        den = xp.asarray([0.3, 1.0, 0.2])

        uflat = xp.asarray(u[:, 0])
        uflat = uflat.reshape((5,))
        tout, yout = scp.signal.dlsim((num, den, 0.5), uflat, t_in)
        return tout, yout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_6(self, xp, scp):
        # zeros-poles-gain representation
        zd = xp.array([0.5, -0.5])
        pd = xp.array([1.j / sqrt(2), -1.j / sqrt(2)])
        k = 1.0

        u = xp.hstack((xp.linspace(0, 4.0, num=5)[:, None],
                       xp.full((5, 1), 0.01),
                       xp.full((5, 1), -0.002)))
        t_in = xp.linspace(0, 2.0, num=5)

        tout, yout = scp.signal.dlsim((zd, pd, k, 0.5), u[:, 0], t_in)
        return tout, yout

    def test_dlsim_7(self):
        # Raise an error for continuous-time systems
        system = signal.lti([1], [1, 1])
        u = cupy.hstack((cupy.linspace(0, 4.0, num=5)[:, None],
                         cupy.full((5, 1), 0.01),
                         cupy.full((5, 1), -0.002)))
        with assert_raises(AttributeError):
            signal.dlsim(system, u)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dstep(self, xp, scp):
        a = xp.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = xp.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = xp.asarray([[0.1, 0.3]])
        d = xp.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        tout, yout = scp.signal.dstep((a, b, c, d, dt), n=10)

        # NB: yout is a 3-tuple, unpack it
        return tout, *yout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dstep_2(self, xp, scp):
        # Check that the other two inputs (tf, zpk) will work as well
        tfin = ([1.0], [1.0, 1.0], 0.5)
        tout, yout = scp.signal.dstep(tfin, n=3)
        return tout, *yout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dstep_3(self, xp, scp):
        tfin = ([1.0], [1.0, 1.0], 0.5)
        zpkin = scp.signal.tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = scp.signal.dstep(zpkin, n=3)
        return tout, *yout

    def test_dstep_4(self):
        # Raise an error for continuous-time systems
        system = signal.lti([1], [1, 1])
        with assert_raises(AttributeError):
            signal.dstep(system)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dimpulse(self, xp, scp):
        a = xp.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = xp.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = xp.asarray([[0.1, 0.3]])
        d = xp.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        tout, yout = scp.signal.dimpulse((a, b, c, d, dt), n=10)
        return tout, *yout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dimpulse_2(self, xp, scp):
        # Check that the other two inputs (tf, zpk) will work as well
        tfin = ([1.0], [1.0, 1.0], 0.5)
        tout, yout = scp.signal.dimpulse(tfin, n=3)
        return tout, *yout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dimpulse_3(self, xp, scp):
        tfin = ([1.0], [1.0, 1.0], 0.5)
        zpkin = scp.signal.tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = scp.signal.dimpulse(zpkin, n=3)
        return tout, *yout

    def test_dimpulse_4(self):
        # Raise an error for continuous-time systems
        system = signal.lti([1], [1, 1])
        assert_raises(AttributeError, signal.dimpulse, system)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_trivial(self, xp, scp):
        a = xp.array([[0.0]])
        b = xp.array([[0.0]])
        c = xp.array([[0.0]])
        d = xp.array([[0.0]])
        n = 5
        u = xp.zeros(n).reshape(-1, 1)
        tout, yout, xout = scp.signal.dlsim((a, b, c, d, 1), u)
        return tout, yout, xout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_simple1d(self, xp, scp):
        a = xp.array([[0.5]])
        b = xp.array([[0.0]])
        c = xp.array([[1.0]])
        d = xp.array([[0.0]])
        n = 5
        u = xp.zeros(n).reshape(-1, 1)
        tout, yout, xout = scp.signal.dlsim((a, b, c, d, 1), u, x0=1)
        return tout, yout, xout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dlsim_simple2d(self, xp, scp):
        lambda1 = 0.5
        lambda2 = 0.25
        a = xp.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        b = xp.array([[0.0],
                      [0.0]])
        c = xp.array([[1.0, 0.0],
                      [0.0, 1.0]])
        d = xp.array([[0.0],
                      [0.0]])
        n = 5
        u = xp.zeros(n).reshape(-1, 1)
        tout, yout, xout = scp.signal.dlsim((a, b, c, d, 1), u, x0=1)
        return tout, yout, xout

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_more_step_and_impulse(self, xp, scp):
        lambda1 = 0.5
        lambda2 = 0.75
        a = xp.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        b = xp.array([[1.0, 0.0],
                      [0.0, 1.0]])
        c = xp.array([[1.0, 1.0]])
        d = xp.array([[0.0, 0.0]])

        n = 10

        # Check a step response.
        ts, ys = scp.signal.dstep((a, b, c, d, 1), n=n)
        return ts, *ys

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_more_step_and_impulse_2(self, xp, scp):
        # Check an impulse response with an initial condition.
        lambda1 = 0.5
        lambda2 = 0.75
        a = xp.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        b = xp.array([[1.0, 0.0],
                      [0.0, 1.0]])
        c = xp.array([[1.0, 1.0]])
        d = xp.array([[0.0, 0.0]])

        n = 10

        x0 = xp.array([1.0, 1.0])
        ti, yi = scp.signal.dimpulse((a, b, c, d, 1), n=n, x0=x0)
        return ti, *yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_more_step_and_impulse_3(self, xp, scp):
        # Check that dt=0.1, n=3 gives 3 time values.
        system = ([1.0], [1.0, -0.5], 0.1)
        t, (y,) = scp.signal.dstep(system, n=3)
        t1, (y1,) = scp.signal.dimpulse(system, n=3)
        return t, y, t1, y1


class TestDlti:
    def test_dlti_instantiation(self):
        # Test that lti can be instantiated.

        dt = 0.05
        # TransferFunction
        s = signal.dlti([1], [-1], dt=dt)
        assert isinstance(s, signal.TransferFunction)
        assert isinstance(s, signal.dlti)
        assert not isinstance(s, signal.lti)
        assert s.dt == dt

        # ZerosPolesGain
        s = signal.dlti(cupy.array([]), cupy.array([-1]), 1, dt=dt)
        assert isinstance(s, signal.ZerosPolesGain)
        assert isinstance(s, signal.dlti)
        assert not isinstance(s, signal.lti)
        assert s.dt == dt

        # StateSpace
        s = signal.dlti([1], [-1], 1, 3, dt=dt)
        assert isinstance(s, signal.StateSpace)
        assert isinstance(s, signal.dlti)
        assert not isinstance(s, signal.lti)
        assert s.dt == dt

        # Number of inputs
        assert_raises(ValueError, signal.dlti, 1)
        assert_raises(ValueError, signal.dlti, 1, 1, 1, 1, 1)


class TestStateSpaceDisc:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        signal.StateSpace(1, 1, 1, 1, dt=dt)
        signal.StateSpace([1], [2], [3], [4], dt=dt)
        signal.StateSpace(cupy.array([[1, 2], [3, 4]]), cupy.array([[1], [2]]),
                          cupy.array([[1, 0]]), cupy.array([[0]]), dt=dt)
        signal.StateSpace(1, 1, 1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = signal.StateSpace(1, 2, 3, 4, dt=0.05)
        assert isinstance(s.to_ss(), signal.StateSpace)
        assert isinstance(s.to_tf(), signal.TransferFunction)
        assert isinstance(s.to_zpk(), signal.ZerosPolesGain)

        # Make sure copies work
        assert signal.StateSpace(s) is not s
        assert s.to_ss() is not s

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_tf() and to_zpk()

        # Getters
        s = signal.StateSpace(1, 1, 1, 1, dt=0.05)
        assert s.poles == cupy.array([1])
        assert s.zeros == cupy.array([0])


class TestTransferFunction:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        signal.TransferFunction(1, 1, dt=dt)
        signal.TransferFunction([1], [2], dt=dt)
        signal.TransferFunction(cupy.array([1]), cupy.array([2]), dt=dt)
        signal.TransferFunction(1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = signal.TransferFunction([1, 0], [1, -1], dt=0.05)
        assert isinstance(s.to_ss(), signal.StateSpace)
        assert isinstance(s.to_tf(), signal.TransferFunction)
        assert isinstance(s.to_zpk(), signal.ZerosPolesGain)

        # Make sure copies work
        assert signal.TransferFunction(s) is not s
        assert s.to_tf() is not s

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_ss() and to_zpk()

        # Getters
        s = signal.TransferFunction([1, 0], [1, -1], dt=0.05)
        assert s.poles == cupy.array([1])
        assert s.zeros == cupy.array([0])


class TestZerosPolesGain:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        signal.ZerosPolesGain(1, 1, 1, dt=dt)
        signal.ZerosPolesGain([1], [2], 1, dt=dt)
        signal.ZerosPolesGain(cupy.array([1]), cupy.array([2]), 1, dt=dt)
        signal.ZerosPolesGain(1, 1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = signal.ZerosPolesGain(1, 2, 3, dt=0.05)
        assert isinstance(s.to_ss(), signal.StateSpace)
        assert isinstance(s.to_tf(), signal.TransferFunction)
        assert isinstance(s.to_zpk(), signal.ZerosPolesGain)

        # Make sure copies work
        assert signal.ZerosPolesGain(s) is not s
        assert s.to_zpk() is not s


@testing.with_requires('scipy')
class Test_dfreqresp:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_manual(self, xp, scp):
        # Test dfreqresp() real part calculation (manual sanity check).
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        system = scp.signal.TransferFunction(1, [1, -0.2], dt=0.1)
        w = [0.1, 1, 10]
        w, H = scp.signal.dfreqresp(system, w=w)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_auto(self, xp, scp):
        # Test dfreqresp() real part calculation.
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        system = scp.signal.TransferFunction(1, [1, -0.2], dt=0.1)
        w = [0.1, 1, 10, 100]
        w, H = scp.signal.dfreqresp(system, w=w)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_freq_range(self, xp, scp):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        # Expected range is from 0.01 to 10.
        system = scp.signal.TransferFunction(1, [1, -0.2], dt=0.1)
        n = 10
        w, H = scp.signal.dfreqresp(system, n=n)
        return w, H

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pole_one(self, xp, scp):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = scp.signal.TransferFunction([1], [1, -1], dt=0.1)
        w, H = scp.signal.dfreqresp(system, n=2)
        return w, H

    def test_error(self):
        # Raise an error for continuous-time systems
        system = signal.lti([1], [1, 1])
        assert_raises(AttributeError, signal.dfreqresp, system)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_from_state_space(self, xp, scp):
        # H(z) = 2 / z^3 - 0.5 * z^2
        system_TF = scp.signal.dlti([2], [1, -0.5, 0, 0])

        A = xp.array([[0.5, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
        B = xp.array([[1, 0, 0]]).T
        C = xp.array([[0, 0, 2]])
        D = 0

        system_SS = scp.signal.dlti(A, B, C, D)
        w = 10.0**xp.arange(-3, 0, .5)
        w1, H1 = scp.signal.dfreqresp(system_TF, w=w)
        w2, H2 = scp.signal.dfreqresp(system_SS, w=w)
        return w1, H1, w2, H2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_from_zpk(self, xp, scp):
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        system_ZPK = scp.signal.dlti([], [0.2], 0.3)
        system_TF = scp.signal.dlti(0.3, [1, -0.2])
        w = [0.1, 1, 10, 100]
        w1, H1 = scp.signal.dfreqresp(system_ZPK, w=w)
        w2, H2 = scp.signal.dfreqresp(system_TF, w=w)
        return w1, H1, w2, H2


@testing.with_requires('scipy')
class Test_bode:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_manual(self, xp, scp):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        dt = 0.1
        system = scp.signal.TransferFunction(0.3, [1, -0.2], dt=dt)
        w = [0.1, 0.5, 1, pi]
        w2, mag, phase = scp.signal.dbode(system, w=w)
        return w2, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_auto(self, xp, scp):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        system = scp.signal.TransferFunction(0.3, [1, -0.2], dt=0.1)
        w = xp.array([0.1, 0.5, 1, pi])
        w2, mag, phase = scp.signal.dbode(system, w=w)
        return w2, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_range(self, xp, scp):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        system = scp.signal.TransferFunction(0.3, [1, -0.2], dt=0.1)
        n = 10
        w, mag, phase = scp.signal.dbode(system, n=n)
        return w, mag, phase

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pole_one(self, xp, scp):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = scp.signal.TransferFunction([1], [1, -1], dt=0.1)
        w, mag, phase = scp.signal.dbode(system, n=2)
        m = mag[xp.isfinite(mag)]   # nan-vs-inf mismatch
        return w, m, phase

    def test_imaginary(self):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        system = signal.TransferFunction([1], [1, 0, 100], dt=0.1)
        signal.dbode(system, n=2)

    def test_error(self):
        # Raise an error for continuous-time systems
        system = signal.lti([1], [1, 1])
        assert_raises(AttributeError, signal.dbode, system)


@testing.with_requires('scipy')
class TestTransferFunctionZConversion:
    """Test private conversions between 'z' and 'z**-1' polynomials."""

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_full(self, xp, scp):
        # Numerator and denominator same order
        num = [2, 3, 4]
        den = [5, 6, 7]
        num2, den2 = scp.signal.TransferFunction._z_to_zinv(num, den)
        num3, den3 = scp.signal.TransferFunction._zinv_to_z(num, den)

        # convert to 1D arrays to help the comparator
        return xp.atleast_1d(num2, den2, num3, den3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_numerator(self, xp, scp):
        # Numerator lower order than denominator
        num = [2, 3]
        den = [5, 6, 7]
        num2, den2 = scp.signal.TransferFunction._z_to_zinv(num, den)
        num3, den3 = scp.signal.TransferFunction._zinv_to_z(num, den)
        return xp.atleast_1d(num2, den2, num3, den3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_denominator(self, xp, scp):
        # Numerator higher order than denominator
        num = [2, 3, 4]
        den = [5, 6]
        num2, den2 = scp.signal.TransferFunction._z_to_zinv(num, den)
        num3, den3 = scp.signal.TransferFunction._zinv_to_z(num, den)
        return xp.atleast_1d(num2, den2, num3, den3)

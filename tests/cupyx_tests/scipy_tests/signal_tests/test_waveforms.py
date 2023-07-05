
import pytest

import cupy
from cupy import testing
import cupyx.scipy.signal  # NOQA

import numpy as np

try:
    import scipy.signal
except ImportError:
    pass


@testing.with_requires('scipy')
class TestChirp:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_linear_at_zero(self, xp, scp):
        w = scp.signal.chirp(
            t=0.0, f0=1.0, f1=2.0, t1=1.0, method='linear')
        return w

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_quadratic_at_zero(self, xp, scp):
        w = scp.signal.chirp(
            t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic')
        return w

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_quadratic_at_zero2(self, xp, scp):
        w = scp.signal.chirp(
            t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic', vertex_zero=False)
        return w

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_logarithmic_at_zero(self, xp, scp):
        w = scp.signal.chirp(
            t=0, f0=1.0, f1=2.0, t1=1.0, method='logarithmic')
        return w

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_hyperbolic_at_zero(self, xp, scp):
        w = scp.signal.chirp(
            t=0, f0=10.0, f1=1.0, t1=1.0, method='hyperbolic')
        return w

    def test_hyperbolic_zero_freq(self):
        # f0=0 or f1=0 must raise a ValueError.
        method = 'hyperbolic'
        t1 = 1.0

        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            t = xp.linspace(0, t1, 5)

            with pytest.raises(ValueError):
                scp.signal.chirp(t, 0, t1, 1, method)

            with pytest.raises(ValueError):
                scp.signal.chirp(t, 1, t1, 0, method)

    def test_unknown_method(self):
        method = "foo"
        f0 = 10.0
        f1 = 20.0
        t1 = 1.0

        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            t = xp.linspace(0, t1, 10)

            with pytest.raises(ValueError):
                scp.signal.chirp(t, f0, t1, f1, method)

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_t1(self, xp, scp):
        f0 = 10.0
        f1 = 20.0
        t = xp.linspace(-1, 1, 11)
        t1 = 3.0
        float_result = scp.signal.chirp(t, f0, t1, f1)
        t1 = 3
        int_result = scp.signal.chirp(t, f0, t1, f1)
        return float_result, int_result

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_f0(self, xp, scp):
        f1 = 20.0
        t1 = 3.0
        t = xp.linspace(-1, 1, 11)
        f0 = 10.0
        float_result = scp.signal.chirp(t, f0, t1, f1)
        f0 = 10
        int_result = scp.signal.chirp(t, f0, t1, f1)
        return float_result, int_result

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_f1(self, xp, scp):
        f0 = 10.0
        t1 = 3.0
        t = xp.linspace(-1, 1, 11)
        f1 = 20.0
        float_result = scp.signal.chirp(t, f0, t1, f1)
        f1 = 20
        int_result = scp.signal.chirp(t, f0, t1, f1)
        return float_result, int_result

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_all(self, xp, scp):
        f0 = 10
        t1 = 3
        f1 = 20
        t = xp.linspace(-1, 1, 11)
        float_result = scp.signal.chirp(
            t, float(f0), float(t1), float(f1))
        int_result = scp.signal.chirp(t, f0, t1, f1)
        return float_result, int_result


@testing.with_requires('scipy')
class TestGaussPulse:
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_fc(self, xp, scp):
        float_result = scp.signal.gausspulse('cutoff', fc=1000.0)
        int_result = scp.signal.gausspulse('cutoff', fc=1000)
        return float_result, int_result

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_bw(self, xp, scp):
        float_result = scp.signal.gausspulse('cutoff', bw=1.0)
        int_result = scp.signal.gausspulse('cutoff', bw=1)
        return float_result, int_result

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_bwr(self, xp, scp):
        float_result = scp.signal.gausspulse('cutoff', bwr=-6.0)
        int_result = scp.signal.gausspulse('cutoff', bwr=-6)
        return float_result, int_result

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_integer_tpr(self, xp, scp):
        float_result = scp.signal.gausspulse('cutoff', tpr=-60.0)
        int_result = scp.signal.gausspulse('cutoff', tpr=-60)
        return float_result, int_result


@testing.with_requires('scipy')
class TestUnitImpulse:
    @pytest.mark.parametrize('size', [7, (3, 3)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_no_index(self, size, xp, scp):
        return scp.signal.unit_impulse(size)

    @pytest.mark.parametrize('args', [(10, 3), ((3, 3), (1, 1)), ((4, 4), 2)])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_index(self, args, xp, scp):
        return scp.signal.unit_impulse(*args)

    @pytest.mark.parametrize('size', [(3, 3), 9])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_mid(self, size, xp, scp):
        return scp.signal.unit_impulse(size, 'mid')

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_dtype(self, xp, scp):
        imp1 = scp.signal.unit_impulse(7)
        imp2 = scp.signal.unit_impulse(5, 3, dtype=int)
        imp3 = scp.signal.unit_impulse((5, 2), (3, 1), dtype=complex)
        return imp1, imp2, imp3


@testing.with_requires('scipy')
class TestSawtooth:
    @pytest.mark.parametrize('width', [1.0, 0.5, 3.0])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sawtooth(self, width, xp, scp):
        t = xp.linspace(0, 1, 500)
        return scp.signal.sawtooth(t, width)


@testing.with_requires('scipy')
class TestSquare:
    @pytest.mark.parametrize('duty', [1.0, 0.5, 3.0])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_square(self, duty, xp, scp):
        t = xp.linspace(0, 1, 500)
        return scp.signal.square(t, duty)

from __future__ import annotations


from unittest import mock

import pytest

import cupy
from cupy import testing
import cupyx.scipy.signal  # NOQA
from cupyx.scipy.signal import _waveforms

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

    @pytest.mark.parametrize('method, f0, f1, vertex_zero', [
        ('linear', -3.0, 3.0, True),
        ('lin', -3.0, 3.0, True),
        ('li', -3.0, 3.0, True),
        ('quadratic', 1.0, 3.0, True),
        ('quadratic', 1.0, 3.0, False),
        ('quad', 1.0, 3.0, True),
        ('q', 1.0, 3.0, True),
        ('logarithmic', 1.0, 3.0, True),
        ('logarithmic', 1.0, 1.0, True),
        ('log', 1.0, 3.0, True),
        ('lo', 1.0, 3.0, True),
        ('hyperbolic', 3.0, 1.0, True),
        ('hyperbolic', 1.0, 1.0, True),
        ('hyp', 3.0, 1.0, True),
    ])
    @pytest.mark.parametrize('scalar', [False, True])
    @testing.with_requires('scipy>=1.15.0')
    @testing.for_dtypes([np.float32, np.float64])
    @testing.numpy_cupy_allclose(
        scipy_name='scp',
        rtol={np.complex64: 1e-5, 'default': 1e-6},
        atol={np.complex64: 1e-5, 'default': 1e-6})
    def test_complex(
            self, method, f0, f1, vertex_zero, scalar, xp, scp, dtype):
        t = xp.asarray(0.25, dtype=dtype) if scalar else xp.linspace(
            -0.25, 1.0, 101, dtype=dtype)
        result = scp.signal.chirp(
            t, f0, 1.0, f1, method=method, phi=37.0,
            vertex_zero=vertex_zero, complex=True)
        if xp is np:
            # SciPy computes the phase in float64 even for float32 times.
            # Compare its reference at CuPy's corresponding complex precision.
            result = result.astype(np.result_type(dtype, 1j), copy=False)
        return result

    @pytest.mark.parametrize('method, vertex_zero', [
        ('linear', True),
        ('quadratic', True),
        ('quadratic', False),
        ('logarithmic', True),
        ('hyperbolic', True),
    ])
    @pytest.mark.parametrize('phi', [0.0, 37.0])
    @pytest.mark.parametrize('dtype, expected_dtype, tol', [
        (cupy.float32, cupy.complex64, 1e-5),
        (cupy.float64, cupy.complex128, 1e-12),
        (cupy.int64, cupy.complex128, 1e-12),
    ])
    def test_complex_properties(
            self, method, vertex_zero, phi, dtype, expected_dtype, tol):
        t = cupy.linspace(0.0, 1.0, 101, dtype=dtype)
        actual = cupyx.scipy.signal.chirp(
            t, 1.0, 1.0, 3.0, method=method, phi=phi,
            vertex_zero=vertex_zero, complex=True)
        expected_real = cupyx.scipy.signal.chirp(
            t, 1.0, 1.0, 3.0, method=method, phi=phi,
            vertex_zero=vertex_zero)

        assert actual.dtype == expected_dtype
        assert expected_real.dtype == actual.real.dtype
        testing.assert_allclose(
            actual.real, expected_real, rtol=tol, atol=tol)
        testing.assert_allclose(
            cupy.abs(actual), cupy.ones_like(expected_real),
            rtol=tol, atol=tol)

    @pytest.mark.parametrize('method, kernel_name, vertex_zero', [
        ('linear', '_chirp_phase_lin_kernel', True),
        ('quadratic', '_chirp_phase_quad_kernel', True),
        ('quadratic', '_chirp_phase_quad_kernel', False),
        ('logarithmic', '_chirp_phase_log_kernel', True),
        ('hyperbolic', '_chirp_phase_hyp_kernel', True),
    ])
    @pytest.mark.parametrize('complex', [False, True])
    @pytest.mark.parametrize('scalar', [False, True])
    @testing.for_dtypes([np.float32, np.float64])
    def test_single_output_and_kernel(
            self, method, kernel_name, vertex_zero, complex, scalar, dtype):
        t = cupy.asarray(0.25, dtype=dtype) if scalar else cupy.linspace(
            0.0, 1.0, 101, dtype=dtype)
        kernel = getattr(_waveforms, kernel_name)
        with mock.patch.object(
                cupy, 'empty_like', wraps=cupy.empty_like) as alloc:
            with mock.patch.object(
                    _waveforms, kernel_name, wraps=kernel) as launch:
                actual = cupyx.scipy.signal.chirp(
                    t, 1.0, 1.0, 3.0, method=method, phi=37.0,
                    vertex_zero=vertex_zero, complex=complex)

        expected_dtype = cupy.result_type(dtype, 1j) if complex else dtype
        assert actual.dtype == expected_dtype
        assert actual.shape == t.shape
        alloc.assert_called_once_with(t, dtype=expected_dtype)
        launch.assert_called_once()
        assert launch.call_args.args[0] is t
        assert launch.call_args.args[-1] is actual

    @pytest.mark.parametrize('method, f0, f1', [
        ('logarithmic', 0.0, 1.0),
        ('hyperbolic', 0.0, 1.0),
    ])
    @testing.with_requires('scipy>=1.15.0')
    def test_complex_invalid_frequency(self, method, f0, f1):
        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            t = xp.linspace(0.0, 1.0, 5)
            with pytest.raises(ValueError):
                scp.signal.chirp(
                    t, f0, 1.0, f1, method=method, complex=True)

    @testing.with_requires('scipy>=1.15.0')
    def test_complex_keyword_only(self):
        for xp, scp in [(cupy, cupyx.scipy), (np, scipy)]:
            t = xp.linspace(0.0, 1.0, 5)
            with pytest.raises(TypeError):
                scp.signal.chirp(
                    t, 1.0, 1.0, 3.0, 'linear', 0.0, True, True)

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


@testing.slow
@pytest.mark.thread_unsafe(reason='Allocation too large.')
def test_unit_impulse_large_size():
    out = None
    try:
        out = cupyx.scipy.signal.unit_impulse(
            2**31, 2**31 - 1, dtype=cupy.uint8)
        testing.assert_array_equal(out[-2:], [0, 1])
    except MemoryError:
        pytest.skip('out of memory in test.')
    finally:
        del out
        cupy.get_default_memory_pool().free_all_blocks()


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

    @pytest.mark.parametrize('width', [0.0, 0.5, 1.0])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sawtooth_negative_input(self, width, xp, scp):
        t = xp.linspace(-2 * xp.pi, 2 * xp.pi, 500)
        return scp.signal.sawtooth(t, width)


@testing.with_requires('scipy')
class TestSquare:
    @pytest.mark.parametrize('duty', [1.0, 0.5, 3.0])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_square(self, duty, xp, scp):
        t = xp.linspace(0, 1, 500)
        return scp.signal.square(t, duty)

    @pytest.mark.parametrize('duty', [0.0, 0.5, 1.0])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_square_negative_input(self, duty, xp, scp):
        t = xp.linspace(-2 * xp.pi, 2 * xp.pi, 500)
        return scp.signal.square(t, duty)


@testing.with_requires('scipy')
class TestSweepPoly:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_quad1(self, xp, scp):
        p = xp.poly1d([1.0, 0.0, 1.0])
        t = xp.linspace(0, 3.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_const(self, xp, scp):
        p = xp.poly1d(2.0)
        t = xp.linspace(0, 3.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_linear(self, xp, scp):
        p = xp.poly1d([-1.0, 10.0])
        t = xp.linspace(0, 3.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_quad2(self, xp, scp):
        p = xp.poly1d([1.0, 0.0, -2.0])
        t = xp.linspace(0, 3.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_cubic(self, xp, scp):
        p = xp.poly1d([2.0, 1.0, 0.0, -2.0])
        t = xp.linspace(0, 2.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_cubic2(self, xp, scp):
        """Use an array of coefficients instead of a poly1d."""
        p = xp.array([2.0, 1.0, 0.0, -2.0])
        t = xp.linspace(0, 2.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sweep_poly_cubic3(self, xp, scp):
        """Test sweep_poly itsefl, not its phase helper."""
        p = xp.asarray([2.0, 1.0, 0.0, -2.0])
        t = xp.linspace(0, 2.0, 10000)
        arr = scp.signal.sweep_poly(t, p)
        return arr

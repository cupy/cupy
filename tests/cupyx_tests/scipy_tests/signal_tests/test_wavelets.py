
import pytest

from cupy import testing
import cupyx.scipy.signal  # NOQA

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.with_requires('scipy')
class TestWavelets:
    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_qmf(self, xp, scp):
        return scp.signal.qmf([1, 1])

    @pytest.mark.skip(reason='daub is not available on cupyx.scipy.signal')
    @pytest.mark.parametrize('p', list(range(1, 15)))
    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_daub(self, p, xp, scp):
        return scp.signal.daub(p)

    @pytest.mark.skip(reason='cascade is not available on cupyx.scipy.signal')
    @pytest.mark.parametrize('J', list(range(1, 7)))
    @pytest.mark.parametrize('i', list(range(1, 5)))
    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_cascade(self, J, i, xp, scp):
        lpcoef = scp.signal.daub(i)
        x, phi, psi = scp.signal.cascade(lpcoef, J)
        return x, phi, psi

    @pytest.mark.parametrize('args,kwargs', [
        ((50, 4.1), {'complete': True}),
        ((50, 4.1), {'complete': False}),
        ((10, 50), {'complete': False}),
        ((10, 50), {'complete': True}),
        ((3,), {'w': 2, 'complete': True}),
        ((3,), {'w': 2, 'complete': False}),
        ((10000,), {'s': 4, 'complete': True}),
        ((10000,), {'s': 4, 'complete': True}),
        ((10000,), {'s': 4, 'complete': False}),
        ((10000,), {'s': 5, 'w': 3, 'complete': True}),
        ((10000,), {'s': 5, 'w': 3, 'complete': False}),
    ])
    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_morlet(self, args, kwargs, xp, scp):
        return scp.signal.morlet(*args, **kwargs)

    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_morlet2(self, xp, scp):
        return scp.signal.morlet2(1.0, 0.5)

    @pytest.mark.parametrize('length', [5, 11, 15, 51, 101])
    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_morlet2_length(self, length, xp, scp):
        return scp.signal.morlet2(length, 1.0)

    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_morlet2_points(self, xp, scp):
        points = 100
        w = scp.signal.morlet2(points, 2.0)
        y = scp.signal.morlet2(3, s=1/(2*xp.pi), w=2)
        return w, y

    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_ricker(self, xp, scp):
        return scp.signal.ricker(1.0, 1)

    @pytest.mark.parametrize('length', [5, 11, 15, 51, 101])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    @testing.with_requires('scipy<1.12.0')
    def test_ricker_length(self, length, xp, scp):
        return scp.signal.ricker(length, 1.0)

    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_ricker_points(self, xp, scp):
        points = 100
        return scp.signal.ricker(points, 2.0)

    @pytest.mark.parametrize('a', [5, 10, 15, 20, 30])
    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_ricker_zeros(self, a, xp, scp):
        # Check zeros
        points = 99
        return scp.signal.ricker(points, a)

    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_cwt_delta(self, xp, scp):
        widths = [1.0]
        len_data = 100
        test_data = xp.sin(xp.pi * xp.arange(0, len_data) / 10.0)

        def delta_wavelet(s, t):
            return xp.array([1])

        return scp.signal.cwt(test_data, delta_wavelet, widths)

    @testing.with_requires('scipy<1.12.0')
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_cwt_ricker(self, xp, scp):
        len_data = 100
        test_data = xp.sin(xp.pi * xp.arange(0, len_data) / 10.0)
        # Check proper shape on output
        widths = [1, 3, 4, 5, 10]
        return scp.signal.cwt(test_data, scp.signal.ricker, widths)

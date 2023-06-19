# This program is public domain
# Authors: Paul Kienzle, Nadav Horesh
'''
A unit test module for czt.py
'''
from math import pi

import pytest
import cupy
import numpy as np

from cupy import testing
from cupy.testing import assert_allclose

import cupyx.scipy.fft as fft
import cupyx.scipy.signal as signal


def check_czt(x, xp, scp):
    # Check that czt is the equivalent of normal fft
    y1 = scp.signal.czt(x)

    # Check that interpolated czt is the equivalent of normal fft
    y1_ = scp.signal.czt(x, 100*len(x))
    return y1, y1_


def check_zoom_fft(x, xp, scp):
    # Check that zoom_fft is the equivalent of normal fft
    y = scp.fft.fft(x)
    y1 = scp.signal.zoom_fft(x, [0, 2-2./len(y)], endpoint=True)
    y2 = scp.signal.zoom_fft(x, [0, 2])
    return y1, y2


def check_zoom_fft_2(x, xp, scp):
    # Test fn scalar
    y = scp.fft.fft(x)
    y1 = scp.signal.zoom_fft(x, 2-2./len(y), endpoint=True)
    y2 = scp.signal.zoom_fft(x, 2)
    return y1, y2


def check_zoom_fft_3(x, xp, scp):
    # Check that zoom_fft with oversampling is equivalent to zero padding
    over = 10
    yover = scp.fft.fft(x, over*len(x))
    y1 = scp.signal.zoom_fft(
        x, [0, 2-2./len(yover)], m=len(yover), endpoint=True)
    y2 = scp.signal.zoom_fft(x, [0, 2], m=len(yover))

    # Check that zoom_fft works on a subrange
    w = xp.linspace(0, 2-2./len(x), len(x))
    f1, f2 = w[3], w[6]
    y3 = scp.signal.zoom_fft(x, [f1, f2], m=3*over+1, endpoint=True)

    return y1, y2, y3


pw2_ranges = [range(126-31), range(127-31), range(128-31), range(129-31),
              range(130-31), ]
zf_checks = [check_zoom_fft, check_zoom_fft_2, check_zoom_fft_3, check_czt]


def _gen_random_signal():
    lengths = testing.shaped_random((8, 200, 20))
    for x in lengths:
        yield x


@testing.with_requires("scipy >= 1.8.0")
class Test1D:

    @pytest.mark.parametrize('func', zf_checks)
    @pytest.mark.parametrize('x', pw2_ranges)
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_near_pow2(self, xp, scp, x, func):
        x = xp.asarray(x)
        return func(x, xp, scp)

    @pytest.mark.parametrize('func', zf_checks)
    @testing.numpy_cupy_allclose(scipy_name="scp", atol=1e-14)
    def test_gauss(self, xp, scp, func):
        t = xp.linspace(-2, 2, 128)
        x = xp.exp(-t**2/0.01)
        return func(x, xp, scp)

    @pytest.mark.parametrize('func', zf_checks)
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_linear(self, xp, scp, func):
        x = xp.asarray([1, 2, 3, 4, 5, 6, 7])
        return func(x, xp, scp)

    @pytest.mark.parametrize('func', zf_checks)
    @testing.numpy_cupy_allclose(scipy_name="scp", atol=1e-13)
    def test_spikes(self, xp, scp, func):
        t = xp.linspace(0, 1, 128)
        x = xp.sin(2*pi*t*5) + xp.sin(2*pi*t*13)
        return func(x, xp, scp)

    @pytest.mark.parametrize('func', zf_checks)
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sines(self, xp, scp, func):
        x = xp.zeros(100, dtype=complex)
        x[[1, 5, 21]] = 1
        return func(x, xp, scp)

    @pytest.mark.parametrize('func', zf_checks)
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_sines_plus(self, xp, scp, func):
        x = xp.zeros(100, dtype=complex)
        x[[1, 5, 21]] = 1
        x += 1j*xp.linspace(0, 0.5, x.shape[0])
        return func(x, xp, scp)

    @pytest.mark.parametrize('func', zf_checks)
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_random(self, xp, scp, func):
        x = testing.shaped_random((101,), xp, dtype=float)
        # check_zoom_fft(x)
        return func(x, xp, scp)

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_nd(self, xp, scp):
        # Check transform on n-D array input
        x = xp.arange(3*2*28).reshape(3, 2, 28)
        y1 = scp.signal.zoom_fft(x, [0, 2-2./28])
        y2 = scp.signal.zoom_fft(x[2, 0, :], [0, 2-2./28])
        return y1, y2

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_nd_2(self, xp, scp):
        # Check transform on n-D array input
        x = xp.arange(3*2*28).reshape(3, 2, 28)
        y1 = scp.signal.zoom_fft(x, [0, 2], endpoint=False)
        y2 = scp.signal.zoom_fft(x[2, 0, :], [0, 2], endpoint=False)
        return y1, y2

    @pytest.mark.parametrize('func', zf_checks)
    @pytest.mark.parametrize("x", list(_gen_random_signal()))
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_random_signals(self, xp, scp, x, func):
        assert x.shape == (200, 20)
        if xp == np:
            x = x.get()
        return func(x, xp, scp)

    @pytest.mark.parametrize("N", [101, 1009, 10007])
    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_large_prime_lengths(self, xp, scp, N):
        x = testing.shaped_random((N,))
        if xp == np:
            x = x.get()
        return scp.signal.czt(x)


@testing.with_requires("scipy >= 1.8.0")
class TestErrors:

    @pytest.mark.parametrize('size', [0, -5, 3.5, 4.0])
    def test_nonsense_size(self, size):
        # Numpy and Scipy fft() give ValueError for 0 output size
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.CZT(size, 3)
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.ZoomFFT(size, 0.2, 3)
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.CZT(3, size)
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.ZoomFFT(3, 0.2, size)
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.czt([1, 2, 3], size)
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.zoom_fft([1, 2, 3], 0.2, size)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match='2-length sequence'):
            signal.ZoomFFT(100, [1, 2, 3])

    @pytest.mark.parametrize('m', [0, -11, 5.5, 4.0])
    def test_czt_points_errors(self, m):
        # Invalid number of points
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.czt_points(m)

    def test_empty_input(self):
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.czt([])
        with pytest.raises(ValueError, match='Invalid number of CZT'):
            signal.zoom_fft([], 0.5)

    def test_0_rank_input(self):
        with pytest.raises(IndexError, match='tuple index out of range'):
            signal.czt(5)
        with pytest.raises(IndexError, match='tuple index out of range'):
            signal.zoom_fft(5, 0.5)


@testing.with_requires("scipy >= 1.8.0")
class TestCZTPoints:

    @pytest.mark.parametrize("N", [1, 2, 3, 8, 11, 100, 101, 10007])
    @testing.numpy_cupy_allclose(scipy_name="scp", rtol=1e-15)
    def test_points(self, xp, scp, N):
        return scp.signal.czt_points(N)

    @testing.numpy_cupy_allclose(scipy_name="scp", rtol=1e-15)
    def test_points_w(self, xp, scp):
        return scp.signal.czt_points(7, w=1), scp.signal.czt_points(11, w=2)

    @testing.numpy_cupy_allclose(scipy_name="scp", rtol=1e-15)
    def test_points_meth(self, xp, scp):
        func = scp.signal.CZT(12, m=11, w=2., a=1)
        return func.points()

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_int_args(self, xp, scp):
        # Integer argument `a` was producing all 0s
        return (scp.signal.czt([0, 1], m=10, a=2),
                scp.signal.czt_points(11, w=2))


@testing.with_requires("scipy >= 1.8.0")
@pytest.mark.slow
def test_czt_vs_fft():
    cupy.random.seed(123)
    random_lengths = cupy.random.exponential(100000, size=10).astype('int')
    for n in random_lengths:
        a = cupy.random.randn(int(n))
        assert_allclose(signal.czt(a), fft.fft(a), rtol=1e-11)


@pytest.mark.parametrize('impulse', ([0, 0, 1], [0, 0, 1, 0, 0],
                                     cupy.concatenate((cupy.array([0, 0, 1]),
                                                       cupy.zeros(100)))))
@pytest.mark.parametrize('m', (1, 3, 5, 8, 101, 1021))
@pytest.mark.parametrize('a', (1, 2, 0.5, 1.1))
# Step that tests away from the unit circle, but not so far it explodes from
# numerical error
@pytest.mark.parametrize('w', (None, 0.98534 + 0.17055j))
def test_czt_math(impulse, m, w, a):
    # z-transform of an impulse is 1 everywhere
    assert_allclose(signal.czt(impulse[2:], m=m, w=w, a=a),
                    cupy.ones(m), rtol=1e-10)

    # z-transform of a delayed impulse is z**-1
    assert_allclose(signal.czt(impulse[1:], m=m, w=w, a=a),
                    signal.czt_points(m=m, w=w, a=a)**-1, rtol=1e-10)

    # z-transform of a 2-delayed impulse is z**-2
    assert_allclose(signal.czt(impulse, m=m, w=w, a=a),
                    signal.czt_points(m=m, w=w, a=a)**-2, rtol=1e-10)

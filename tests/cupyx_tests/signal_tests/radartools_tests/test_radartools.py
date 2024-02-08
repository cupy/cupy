import numpy
import pytest
import scipy

import cupy
from cupy import testing
from cupyx import signal


def _numpy_pulse_preprocess(x, normalize, window):
    if window is not None:
        n = x.shape[-1]
        if callable(window):
            x *= window(numpy.fft.fftfreq(n))
        else:
            from scipy.signal import get_window
            x *= get_window(window, n, False)

    if normalize:
        x /= numpy.linalg.norm(x)

    return x


def _numpy_pulse_doppler(x, window):
    xT = _numpy_pulse_preprocess(x.T, False, window)
    fft_x = numpy.fft.fft(xT, xT.shape[1]).T
    return fft_x.astype(x.dtype.char.upper())


def _numpy_pulse_compression(x, t, normalize, window):
    n = x.shape[1]
    dtype = numpy.result_type(x, t)
    t = _numpy_pulse_preprocess(t, normalize, window)
    fft_x = numpy.fft.fft(x, n)
    fft_t = numpy.fft.fft(t, n)
    out = numpy.fft.ifft(fft_x * fft_t.conj(), n)
    if dtype.kind != 'c':
        out = out.real
    return out.astype(dtype)


tol = {
    numpy.float32: 2e-3,
    numpy.float64: 1e-7,
    numpy.complex64: 2e-3,
    numpy.complex128: 1e-7,
}


@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('window', [None, 'hamming', numpy.negative])
@testing.for_dtypes('fdFD')
@testing.numpy_cupy_allclose(rtol=tol, contiguous_check=False)
def test_pulse_compression(xp, normalize, window, dtype):
    x = testing.shaped_random((8, 700), xp=xp, dtype=dtype)
    template = testing.shaped_random((100,), xp=xp, dtype=dtype)

    if xp is cupy:
        return signal.pulse_compression(x, template, normalize, window)
    else:
        assert xp is numpy
        return _numpy_pulse_compression(x, template, normalize, window)


@pytest.mark.parametrize('window', [None, 'hamming', numpy.negative])
@testing.for_dtypes('fdFD')
@testing.numpy_cupy_allclose(rtol=tol, contiguous_check=False)
def test_pulse_doppler(xp, window, dtype):
    x = testing.shaped_random((8, 700), xp=xp, dtype=dtype)

    if xp is cupy:
        return signal.pulse_doppler(x, window)
    else:
        assert xp is numpy
        return _numpy_pulse_doppler(x, window)


@testing.for_all_dtypes()
def test_cfar_alpha(dtype):
    N = 128
    pfa = testing.shaped_random((128, 128), xp=cupy, dtype=dtype)
    gpu = signal.cfar_alpha(pfa, N)
    cpu = N * (pfa ** (-1.0 / N) - 1)
    cupy.testing.assert_allclose(gpu, cpu)


@pytest.mark.parametrize(
    "size,gc,rc", [(100, 1, 5), (11, 2, 3), (100, 10, 20)])
@testing.for_float_dtypes(no_float16=True)
@testing.numpy_cupy_allclose(rtol=2e-6, type_check=False)
def test_ca_cfar1d(xp, dtype, size, gc, rc):
    array = testing.shaped_random((size,), xp=xp, dtype=dtype)
    if xp is cupy:
        return signal.ca_cfar(array, gc, rc)
    else:
        assert xp is numpy
        weight = numpy.ones(((rc + gc) * 2 + 1,), dtype=dtype)
        weight[rc:-rc] = 0
        alpha = numpy.zeros((size,), dtype=dtype)
        alpha[gc+rc:-gc-rc] = signal.cfar_alpha(1e-3, 2 * rc)
        out = scipy.ndimage.convolve1d(array, weight) * alpha / (2 * rc)
        return out, array - out > 0


@pytest.mark.parametrize(
    "size,gc,rc", [(1, 1, 1), (10, 2, 3), (10, 0, 5), (10, 5, 0)])
def test_ca_cfar1d_failures(size, gc, rc):
    with pytest.raises(ValueError):
        _, _ = signal.ca_cfar(cupy.zeros(size), gc, rc)


@pytest.mark.parametrize(
    "shape,gc,rc", [((10, 10), (1, 1), (2, 2)), ((10, 100), (1, 10), (2, 20))])
@testing.for_float_dtypes(no_float16=True)
@testing.numpy_cupy_allclose(rtol=2e-6, type_check=False)
def test_ca_cfar2d(xp, dtype, shape, gc, rc):
    array = testing.shaped_random(shape, xp=xp, dtype=dtype)
    if xp is cupy:
        return signal.ca_cfar(array, gc, rc)
    else:
        assert xp is numpy
        rcx, rcy = rc
        gcx, gcy = gc
        weight = numpy.ones(
            ((rcx + gcx) * 2 + 1, (rcy + gcy) * 2 + 1), dtype=dtype)
        weight[rcx:-rcx, rcy:-rcy] = 0
        alpha = numpy.zeros(shape, dtype=dtype)
        N = weight.size - (2 * gcx + 1) * (2 * gcy + 1)
        alpha[gcx+rcx:-gcx-rcx, gcy+rcy:-gcy-rcy] = signal.cfar_alpha(1e-3, N)
        out = scipy.ndimage.convolve(array, weight) * alpha / N
        return out, array - out > 0


@pytest.mark.parametrize(
    "shape,gc,rc", [((3, 3), (1, 2), (1, 10)),
                    ((3, 3), (1, 1), (10, 1)),
                    ((5, 5), (3, 3), (3, 3))])
def test_ca_cfar2d_failures(shape, gc, rc):
    with pytest.raises(ValueError):
        _, _ = signal.ca_cfar(cupy.zeros(shape), gc, rc)

import numpy
import pytest

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

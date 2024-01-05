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


class TestCaCfar:
    @pytest.mark.parametrize("length, guard_cells, reference_cells",
                             [(100, 1, 5), (11, 2, 3), (100, 10, 20)])
    class TestOneD:
        def expected(self, length, guard_cells, reference_cells):
            out = cupy.zeros(length)
            N = 2 * reference_cells
            alpha = signal.cfar_alpha(0.001, N)
            out[guard_cells + reference_cells:
                -guard_cells - reference_cells] = cupy.ones(length -
                                                            2 * guard_cells -
                                                            2 * reference_cells)
            return (alpha * out)

        def test_1d_ones(self, length, guard_cells, reference_cells):
            array = cupy.ones(length)
            mask, _ = signal.ca_cfar(array, guard_cells, reference_cells)
            key = self.expected(length, guard_cells, reference_cells)
            testing.numpy_cupy_array_equal(mask, key)

    @pytest.mark.parametrize("length, gc, rc",
                             [(1, 1, 1), (10, 2, 3), (10, 0, 5),
                              (10, 5, 0)])
    class TestFailuresOneD:
        def test_1d_failures(self, length, gc, rc):
            with pytest.raises(ValueError):
                _, _ = signal.ca_cfar(cupy.zeros(length), gc, rc)

    @pytest.mark.parametrize("shape, gc, rc",
                             [((10, 10), (1, 1), (2, 2)), ((10, 100), (1, 10),
                                                           (2, 20))])
    class TestTwoD:
        def expected(self, shape, gc, rc):
            out = cupy.zeros(shape)
            N = 2 * rc[0] * (2 * rc[1] + 2 * gc[1] + 1)
            N += 2 * (2 * gc[0] + 1) * rc[1]
            alpha = signal.cfar_alpha(.001, N)
            out[gc[0] + rc[0]: -gc[0] - rc[0], gc[1] + rc[1]:
                - gc[1] - rc[1]] = cupy.ones((shape[0] - 2 * gc[0] - 2 * rc[0],
                                              shape[1] - 2 * gc[1] - 2 * rc[1]))
            return (alpha * out)

        def test_2d_ones(self, shape, gc, rc):
            array = cupy.ones(shape)
            mask, _ = signal.ca_cfar(array, gc, rc)
            key = self.expected(shape, gc, rc)
            testing.numpy_cupy_array_equal(mask, key)

    @pytest.mark.parametrize("shape, gc, rc",
                             [((3, 3), (1, 2), (1, 10)),
                              ((3, 3), (1, 1), (10, 1)),
                              ((5, 5), (3, 3), (3, 3))])
    class TestFailuresTwoD:
        def test_2d_failures(self, shape, gc, rc):
            with pytest.raises(ValueError):
                _, _ = signal.ca_cfar(cupy.zeros(shape), gc, rc)

    @pytest.mark.parametrize("shape, gc, rc, points",
                             [(10, 1, 1, (6,)), (100, 10, 20, (34, 67)),
                              ((100, 200), (5, 10), (10, 20), [(31, 45),
                                                               (50, 111)])])
    class TestDetection:
        def test_point_detection(self, shape, gc, rc, points):
            '''
            Placing points too close together can yield unexpected results.
            '''
            array = cupy.zeros(shape)
            for point in points:
                array[point] = 1e3
            threshold, detections = signal.ca_cfar(array, gc, rc)
            key = array - threshold
            f = numpy.vectorize(lambda x: True if x > 0 else False)
            key = f(key.get())
            testing.numpy_cupy_array_equal(detections, key)

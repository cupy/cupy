import pytest

import cupy
import numpy
from cupyx import signal


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
def test_pulse_compression(dtype):
    num_pulses = 128
    num_samples_per_pulse = 9000
    template_length = 1000

    shape = num_pulses, num_samples_per_pulse
    x = cupy.random.randn(*shape, dtype=dtype) + \
        1j * cupy.random.rand(*shape, dtype=dtype)
    template = cupy.random.randn(template_length, dtype=dtype) + \
        1j * cupy.random.randn(template_length, dtype=dtype)

    signal.pulse_compression(x, template, normalize=True, window='hamming')


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
def test_pulse_doppler(dtype):
    num_pulses = 128
    num_samples_per_pulse = 9000

    shape = num_pulses, num_samples_per_pulse
    x = cupy.random.randn(*shape, dtype=dtype) + \
        1j * cupy.random.rand(*shape, dtype=dtype)

    signal.pulse_doppler(x, window='hamming')


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
def test_cfar_alpha(dtype):
    N = 128
    pfa = numpy.random.rand(128, 128)
    gpu = signal.cfar_alpha(cupy.array(pfa), N)
    cpu = N * (pfa ** (-1.0 / N) - 1)
    cupy.testing.assert_allclose(gpu, cpu)

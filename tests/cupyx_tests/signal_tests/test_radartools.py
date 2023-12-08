import pytest

import cupy
from cupy import testing
from cupyx import signal


class TestPulseCompression:

    @pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
    def test_pulse_compression(self, dtype):
        num_pulses = 128
        num_samples_per_pulse = 9000
        template_length = 1000
        
        x = cupy.random.randn(num_pulses, num_samples_per_pulse, dtype=dtype) + 1j * cupy.random.randn(num_pulses, num_samples_per_pulse, dtype=dtype)
        template = cupy.random.randn(template_length, dtype=dtype) + 1j * cupy.random.randn(template_length, dtype=dtype)

        signal.pulse_compression(x, template, normalize=True, window='hamming')

import cupy
from cupy import testing
from cupyx import signal


class TestPulseCompression:

    def test_pulse_compression(self):
        num_pulses = 128
        num_samples_per_pulse = 9000
        template_length = 1000
        
        x = cupy.random.randn(num_pulses, num_samples_per_pulse) + 1j * cupy.random.randn(num_pulses, num_samples_per_pulse)
        template = cupy.random.randn(template_length) + 1j * cupy.random.randn(template_length)

        signal.pulse_compression(x, template, normalize=True, window='hamming')

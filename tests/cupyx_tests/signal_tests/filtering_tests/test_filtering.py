import pytest

import numpy
import scipy

import cupy
import cupyx.signal
import cupyx.scipy.signal
from cupy import testing


def linspace_data_gen(start, stop, n, endpoint=False, dtype=numpy.float64):
    cpu_time = numpy.linspace(start, stop, n, endpoint, dtype=dtype)
    cpu_sig = numpy.cos(-(cpu_time**2) / 6.0)
    gpu_sig = cupy.asarray(cpu_sig)
    return cpu_sig, gpu_sig


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
@pytest.mark.parametrize("num_samps", [2**14, 2**18])
@pytest.mark.parametrize("filter_len", [8, 32, 128])
def test_firfilter(dtype, num_samps, filter_len):
    cpu_sig, gpu_sig = linspace_data_gen(0, 10, num_samps, endpoint=False)
    cpu_filter, _ = scipy.signal.butter(filter_len, 0.5)
    gpu_filter = cupy.asarray(cpu_filter)
    cpu_output = scipy.signal.lfilter(cpu_filter, 1, cpu_sig)
    gpu_output = cupyx.signal.firfilter(gpu_filter, gpu_sig)
    testing.assert_allclose(gpu_output, cpu_output, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
@pytest.mark.parametrize("filter_len", [8, 32, 128])
def test_firfilter_zi(dtype, filter_len):
    cpu_filter, _ = scipy.signal.butter(filter_len, 0.5)
    gpu_filter = cupy.asarray(cpu_filter, dtype=dtype)
    gpu_output = cupyx.signal.firfilter_zi(gpu_filter)
    cpu_output = scipy.signal.lfilter_zi(cpu_filter, 1.0)
    # Values are big so a big atol is ok
    testing.assert_allclose(gpu_output, cpu_output, atol=5e-1)
    assert cupy.real(gpu_output).dtype == dtype


@pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
@pytest.mark.parametrize("num_samps", [2**14, 2**18])
@pytest.mark.parametrize("filter_len", [8, 32, 128])
@pytest.mark.parametrize("padtype", ["odd", "even", "constant"])
def test_firfilter2(dtype, num_samps, filter_len, padtype):
    cpu_sig, gpu_sig = linspace_data_gen(0, 10, num_samps, endpoint=False)
    cpu_filter, _ = scipy.signal.butter(filter_len, 0.5)
    gpu_filter = cupy.asarray(cpu_filter)
    cpu_output = scipy.signal.filtfilt(cpu_filter, 1, cpu_sig, padtype=padtype)
    gpu_output = cupyx.signal.firfilter2(gpu_filter, gpu_sig, padtype=padtype)
    testing.assert_allclose(gpu_output, cpu_output, atol=1e-3, rtol=1e-3)


def freq_shift_cpu(x, freq, fs):
    """
    Frequency shift signal by freq at fs sample rate
    Parameters
    ----------
    x : array_like, complex valued
        The data to be shifted.
    freq : float
        Shift by this many (Hz)
    fs : float
        Sampling rate of the signal
    domain : string
        freq or time
    """
    x = numpy.asarray(x)
    return x * numpy.exp(-1j * 2 * numpy.pi * freq / fs * numpy.arange(x.size))


@pytest.mark.parametrize("dtype", [cupy.float64, cupy.complex128])
@pytest.mark.parametrize("num_samps", [2**8])
@pytest.mark.parametrize("freq", numpy.fft.fftfreq(10, 0.1))
@pytest.mark.parametrize("fs", [0.3])
def test_freq_shift(dtype, num_samps, freq, fs):
    cpu_output = freq_shift_cpu(freq, fs, num_samps)
    gpu_output = cupyx.signal.freq_shift(freq, fs, num_samps)
    testing.assert_allclose(gpu_output, cpu_output)

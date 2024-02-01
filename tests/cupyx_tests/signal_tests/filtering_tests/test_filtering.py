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


def channelize_poly_cpu(x, h, n_chans):
    """
    Polyphase channelize signal into n channels
    Parameters
    ----------
    x : array_like
        The input data to be channelized
    h : array_like
        The 1-D input filter; will be split into n
        channels of int number of taps
    n_chans : int
        Number of channels for channelizer
    Returns
    ----------
    yy : channelized output matrix
    Notes
    ----------
    Currently only supports simple channelizer where channel
    spacing is equivalent to the number of channels used
    """

    # number of taps in each h_n filter
    n_taps = int(len(h) / n_chans)

    # number of outputs
    n_pts = int(len(x) / n_chans)

    dtype = cupy.promote_types(x.dtype, h.dtype)

    # order F if input from MATLAB
    h = numpy.conj(numpy.reshape(h.astype(dtype=dtype), (n_taps, n_chans)).T)

    vv = numpy.empty(n_chans, dtype=dtype)

    if x.dtype == numpy.float32 or x.dtype == numpy.complex64:
        yy = numpy.empty((n_chans, n_pts), dtype=numpy.complex64)
    elif x.dtype == numpy.float64 or x.dtype == numpy.complex128:
        yy = numpy.empty((n_chans, n_pts), dtype=numpy.complex128)

    reg = numpy.zeros((n_chans, n_taps), dtype=dtype)

    # instead of n_chans here, this could be channel separation
    for i, nn in enumerate(range(0, len(x), n_chans)):
        reg[:, 1:n_taps] = reg[:, 0: (n_taps - 1)]
        reg[:, 0] = numpy.conj(numpy.flipud(x[nn: (nn + n_chans)]))
        for mm in range(n_chans):
            vv[mm] = numpy.dot(reg[mm, :], numpy.atleast_2d(h[mm, :]).T)[0]

        yy[:, i] = numpy.conj(scipy.fft.fft(vv))

    return yy


@pytest.mark.skipif(
    cupy.cuda.runtime.runtimeGetVersion() < 11040,
    reason='Requires CUDA 11.4 or greater')
@pytest.mark.parametrize(
    "dtype", [cupy.float32, cupy.float64, cupy.complex64, cupy.complex128]
)
@pytest.mark.parametrize("num_samps", [2**12])
@pytest.mark.parametrize("filt_samps", [2048])
@pytest.mark.parametrize("n_chan", [64, 128, 256])
def test_channelize_poly(dtype, num_samps, filt_samps, n_chan):
    cpu_sig = testing.shaped_random((num_samps,), xp=numpy, dtype=dtype)
    gpu_sig = testing.shaped_random((num_samps,), xp=cupy, dtype=dtype)
    cpu_filt = testing.shaped_random((filt_samps,), xp=numpy, dtype=dtype)
    gpu_filt = testing.shaped_random((filt_samps,), xp=cupy, dtype=dtype)
    cpu_output = channelize_poly_cpu(cpu_sig, cpu_filt, n_chan)
    gpu_output = cupyx.signal.channelize_poly(gpu_sig, gpu_filt, n_chan)
    testing.assert_allclose(gpu_output, cpu_output, atol=5e-3, rtol=5e-3)

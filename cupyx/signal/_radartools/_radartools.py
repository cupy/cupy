"""
Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import cupy as cupy
from cupyx.scipy.signal import windows


def pulse_compression(x, template, normalize=False, window=None, nfft=None):
    """
    Pulse Compression is used to increase the range resolution and SNR
    by performing matched filtering of the transmitted pulse (template)
    with the received signal (x)

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    template : ndarray
        Transmitted signal, assume 1D array

    normalize : bool
        Normalize transmitted signal

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    compressedIQ : ndarray
        Pulse compressed output
    """
    [num_pulses, samples_per_pulse] = x.shape

    if nfft is None:
        nfft = samples_per_pulse

    if window is not None:
        Nx = len(template)
        if callable(window):
            W = window(cupy.fft.fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = windows.get_window(window, Nx, False)

        template = cupy.multiply(template, W)

    if normalize is True:
        template = cupy.divide(template, cupy.linalg.norm(template))

    fft_x = cupy.fft.fft(x, nfft)
    fft_template = cupy.conj(
        cupy.tile(cupy.fft.fft(template, nfft), (num_pulses, 1)))
    compressedIQ = cupy.fft.ifft(cupy.multiply(fft_x, fft_template), nfft)

    return compressedIQ


def pulse_doppler(x, window=None, nfft=None):
    """
    Pulse doppler processing yields a range/doppler data matrix that represents
    moving target data that's separated from clutter. An estimation of the
    doppler shift can also be obtained from pulse doppler processing. FFT taken
    across slow-time (pulse) dimension.

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    pd_dataMatrix : ndarray
        Pulse-doppler output (range/doppler matrix)
    """
    [num_pulses, samples_per_pulse] = x.shape

    if nfft is None:
        nfft = num_pulses

    if window is not None:
        Nx = num_pulses
        if callable(window):
            W = window(cupy.fft.fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = windows.get_window(window, Nx, False)[cupy.newaxis]

        pd_dataMatrix = cupy.fft.fft(
            cupy.multiply(x, cupy.tile(W.T, (1, samples_per_pulse))),
            nfft, axis=0
        )
    else:
        pd_dataMatrix = cupy.fft.fft(x, nfft, axis=0)

    return pd_dataMatrix


def cfar_alpha(pfa, N):
    """
    Computes the value of alpha corresponding to a given probability
    of false alarm and number of reference cells N.

    Parameters
    ----------
    pfa : float
        Probability of false alarm.

    N : int
        Number of reference cells.

    Returns
    -------
    alpha : float
        Alpha value.
    """
    return N * (pfa ** (-1.0 / N) - 1)

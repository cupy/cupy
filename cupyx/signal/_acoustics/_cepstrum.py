# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE


import cupy


_real_cepstrum_kernel = cupy.ElementwiseKernel(
    "T spectrum",
    "T output",
    """
    output = log( abs( spectrum ) );
    """,
    "_real_cepstrum_kernel",
    options=("-std=c++11",),
)


def real_cepstrum(x, n=None, axis=-1):
    r"""
    Calculates the real cepstrum of an input sequence x where the cepstrum is
    defined as the inverse Fourier transform of the log magnitude DFT
    (spectrum) of a signal. It's primarily used for source/speaker separation
    in speech signal processing

    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    n : int
        Size of Fourier Transform; If none, will use length of input array
    axis: int
        Direction for cepstrum calculation

    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cupy.asarray(x)
    spectrum = cupy.fft.fft(x, n=n, axis=axis)
    spectrum = _real_cepstrum_kernel(spectrum)
    return cupy.fft.ifft(spectrum, n=n, axis=axis).real


_complex_cepstrum_kernel = cupy.ElementwiseKernel(
    "C spectrum, raw T unwrapped",
    "C output, T ndelay",
    """
    ndelay = round( unwrapped[center] / M_PI );
    const T temp { unwrapped[i] - ( M_PI * ndelay * i / center ) };

    output = log( abs( spectrum ) ) + C( 0, temp );
    """,
    "_complex_cepstrum_kernel",
    options=("-std=c++11",),
    return_tuple=True,
    loop_prep="const int center { static_cast<int>( 0.5 * \
        ( _ind.size() + 1 ) ) };",
)


def complex_cepstrum(x, n=None, axis=-1):
    r"""
    Calculates the complex cepstrum of a real valued input sequence x
    where the cepstrum is defined as the inverse Fourier transform
    of the log magnitude DFT (spectrum) of a signal. It's primarily
    used for source/speaker separation in speech signal processing.

    The input is altered to have zero-phase at pi radians (180 degrees)
    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    n : int
       Size of Fourier Transform; If none, will use length of input array
    axis: int
        Direction for cepstrum calculation
    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cupy.asarray(x)
    spectrum = cupy.fft.fft(x, n=n, axis=axis)
    unwrapped = cupy.unwrap(cupy.angle(spectrum))
    log_spectrum, ndelay = _complex_cepstrum_kernel(spectrum, unwrapped)
    ceps = cupy.fft.ifft(log_spectrum, n=n, axis=axis).real

    return ceps, ndelay


_inverse_complex_cepstrum_kernel = cupy.ElementwiseKernel(
    "C log_spectrum, int32 ndelay, float64 pi",
    "C spectrum",
    """
    const double wrapped { log_spectrum.imag() + M_PI * ndelay * i / center };

    spectrum = exp( C( log_spectrum.real(), wrapped ) )
    """,
    "_inverse_complex_cepstrum_kernel",
    options=("-std=c++11",),
    loop_prep="const double center { 0.5 * ( _ind.size() + 1 ) };",
)


def inverse_complex_cepstrum(ceps, ndelay):
    r"""Compute the inverse complex cepstrum of a real sequence.
    ceps : ndarray
        Real sequence to compute inverse complex cepstrum of.
    ndelay: int
        The amount of samples of circular delay added to `x`.
    Returns
    -------
    x : ndarray
        The inverse complex cepstrum of the real sequence `ceps`.
    The inverse complex cepstrum is given by
    .. math:: x[n] = F^{-1}\left{\exp(F(c[n]))\right}
    where :math:`c_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    """
    ceps = cupy.asarray(ceps)
    log_spectrum = cupy.fft.fft(ceps)
    spectrum = _inverse_complex_cepstrum_kernel(log_spectrum, ndelay, cupy.pi)
    iceps = cupy.fft.ifft(spectrum).real

    return iceps


_minimum_phase_kernel = cupy.ElementwiseKernel(
    "T ceps",
    "T window",
    """
    if ( !i ) {
        window = ceps;
    } else if ( i < bend ) {
        window = ceps * 2.0;
    } else if ( i == bend ) {
        window = ceps * ( 1 - odd );
    } else {
        window = 0;
    }
    """,
    "_minimum_phase_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend { static_cast<int>( 0.5 * \
                    ( _ind.size() + odd ) ) };",
)


def minimum_phase(x, n=None):
    r"""Compute the minimum phase reconstruction of a real sequence.
    x : ndarray
        Real sequence to compute the minimum phase reconstruction of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.
    Returns
    -------
    m : ndarray
        The minimum phase reconstruction of the real sequence `x`.
    """
    if n is None:
        n = len(x)
    ceps = real_cepstrum(x, n=n)
    window = _minimum_phase_kernel(ceps)
    m = cupy.fft.ifft(cupy.exp(cupy.fft.fft(window))).real

    return m

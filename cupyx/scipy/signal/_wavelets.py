
"""
Wavelet-generating functions.

Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
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

import cupy
import numpy as np

from cupyx.scipy.signal._signaltools import convolve


_qmf_kernel = cupy.ElementwiseKernel(
    "raw T coef",
    "T output",
    """
    const int sign { ( i & 1 ) ? -1 : 1 };
    output = ( coef[_ind.size() - ( i + 1 )] ) * sign;
    """,
    "_qmf_kernel",
    options=("-std=c++11",),
)


def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    hk = cupy.asarray(hk)
    return _qmf_kernel(hk, size=len(hk))


_morlet_kernel = cupy.ElementwiseKernel(
    "float64 w, float64 s, bool complete",
    "complex128 output",
    """
    const double x { start + delta * i };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    if ( complete ) {
        temp -= exp( -0.5 * ( w * w ) );
    }

    output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="const double end { s * 2.0 * M_PI }; \
               const double start { -s * 2.0 * M_PI }; \
               const double delta { ( end - start ) / ( _ind.size() - 1 ) };",
)


def morlet(M, w=5.0, s=1.0, complete=True):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    cupyx.scipy.signal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    """
    return _morlet_kernel(w, s, complete, size=M)


_ricker_kernel = cupy.ElementwiseKernel(
    "float64 a",
    "float64 total",
    """
    const double vec { i - ( _ind.size() - 1.0 ) * 0.5 };
    const double xsq { vec * vec };
    const double mod { 1 - xsq / wsq };
    const double gauss { exp( -xsq / ( 2.0 * wsq ) ) };

    total = A * mod * gauss;
    """,
    "_ricker_kernel",
    options=("-std=c++11",),
    loop_prep="const double A { 2.0 / ( sqrt( 3 * a ) * pow( M_PI, 0.25 ) ) };"
    " const double wsq { a * a };",
)


def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,

    where ``A = 2/sqrt(3a)pi^1/4``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> import cupyx.scipy.signal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = cupyx.scipy.signal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(cupy.asnumpy(vec2))
    >>> plt.show()

    """
    return _ricker_kernel(a, size=int(points))


_morlet2_kernel = cupy.ElementwiseKernel(
    "float64 w, float64 s",
    "complex128 output",
    """
    const double x { ( i - ( _ind.size() - 1.0 ) * 0.5 ) / s };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    output = sqrt( 1 / s ) * temp * exp( -0.5 * ( x * x ) ) *
        pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="",
)


def morlet2(M, s, w=5):
    """
    Complex Morlet wavelet, designed to work with `cwt`.
    Returns the complete version of morlet wavelet, normalised
    according to `s`::

        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)

    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter of the wavelet.
    w : float, optional
        Omega0. Default is 5

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    morlet : Implementation of Morlet wavelet, incompatible with `cwt`

    Notes
    -----
    This function was designed to work with `cwt`. Because `morlet2`
    returns an array of complex numbers, the `dtype` argument of `cwt`
    should be set to `complex128` for best results.

    Note the difference in implementation with `morlet`.
    The fundamental frequency of this wavelet in Hz is given by::

        f = w*fs / (2*s*np.pi)

    where ``fs`` is the sampling rate and `s` is the wavelet width parameter.
    Similarly we can get the wavelet width parameter at ``f``::

        s = w*fs / (2*f*np.pi)

    Examples
    --------
    >>> from cupyx.scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> M = 100
    >>> s = 4.0
    >>> w = 2.0
    >>> wavelet = signal.morlet2(M, s, w)
    >>> plt.plot(abs(wavelet))
    >>> plt.show()

    This example shows basic use of `morlet2` with `cwt` in time-frequency
    analysis:

    >>> from cupyx.scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t, dt = np.linspace(0, 1, 200, retstep=True)
    >>> fs = 1/dt
    >>> w = 6.
    >>> sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
    >>> freq = np.linspace(1, fs/2, 100)
    >>> widths = w*fs / (2*freq*np.pi)
    >>> cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
    >>> plt.pcolormesh(t, freq, np.abs(cwtm),
        cmap='viridis', shading='gouraud')
    >>> plt.show()
    """
    return _morlet2_kernel(w, s, size=int(M))


def cwt(data, wavelet, widths):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = cupyx.scipy.signal.convolve(data, wavelet(length,
                                    width[ii]), mode='same')

    Examples
    --------
    >>> import cupyx.scipy.signal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cupy.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = cupy.cos(2 * cupy.pi * 7 * t) + cupyx.scipy.signal.gausspulse(t - 0.4, fc=2)
    >>> widths = cupy.arange(1, 31)
    >>> cwtmatr = cupyx.scipy.signal.cwt(sig, cupyx.scipy.signal.ricker, widths)
    >>> plt.imshow(abs(cupy.asnumpy(cwtmatr)), extent=[-1, 1, 31, 1],
                   cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())
    >>> plt.show()

    """  # NOQA
    if cupy.asarray(wavelet(1, 1)).dtype.char in "FDG":
        dtype = cupy.complex128
    else:
        dtype = cupy.float64

    output = cupy.empty([len(widths), len(data)], dtype=dtype)

    for ind, width in enumerate(widths):
        N = np.min([10 * int(width), len(data)])
        wavelet_data = cupy.conj(wavelet(N, int(width)))[::-1]
        output[ind, :] = convolve(data, wavelet_data, mode="same")
    return output

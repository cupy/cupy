
"""
Signal sampling functions.

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

from math import gcd

import cupy
from cupyx.scipy.signal._fir_filter_design import firwin
from cupyx.scipy.signal._upfirdn import upfirdn, _output_len, _upfirdn_modes
from cupyx.scipy.signal.windows._windows import get_window


def _design_resample_poly(up, down, window):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `cusignal.get_window` and `cusignal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window)
    return h


def decimate(x, q, n=None, axis=-1, zero_phase=True):
    """
    Downsample the signal after applying an anti-aliasing filter.

    Parameters
    ----------
    x : array_like
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor.
    n : int or array_like, optional
        The order of the filter (1 less than the length for FIR) to calculate,
        or the FIR filter coefficients to employ. Defaults to calculating the
        coefficients for 20 times the downsampling factor.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See Also
    --------
    resample : Resample up or down using the FFT method.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    Only FIR filter types are currently supported in cuSignal.
    """

    x = cupy.asarray(x)

    if isinstance(n, (list, cupy.ndarray)):
        b = cupy.asarray(n)
    else:
        if n is None:
            half_len = 10 * q  # reasonable cutoff for our sinc-like function
            n = 2 * half_len

        b = firwin(n + 1, 1.0 / q, window="hamming")

    sl = [slice(None)] * x.ndim

    if zero_phase:
        y = resample_poly(x, 1, q, axis=axis, window=b)
    else:
        # upfirdn is generally faster than lfilter by a factor equal to the
        # downsampling factor, since it only calculates the needed outputs
        n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
        y = upfirdn(b, x, 1, q, axis)
        sl[axis] = slice(None, n_out, None)

    return y[tuple(sl)]


def resample(x, num, t=None, axis=0, window=None, domain="time"):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the sample positions
        associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:

        ``time``
           Consider the input `x` as time-domain. (Default)
        ``freq``
           Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `cusignal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it represents the old sample positions,
    and the new sample positions will be returned as well as the new
    samples.

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fftpack.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> import cupy as cp
    >>> import cupyx.scipy.signal import resample

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f = resample(y, 100)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'go-', cupy.asnumpy(xnew), \
                cupy.asnumpy(f), '.-', 10, cupy.asnumpy(y[0]), 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()
    """
    x = cupy.asarray(x)
    Nx = x.shape[axis]

    if domain == "time":
        X = cupy.fft.fft(x, axis=axis)
    elif domain == "freq":
        X = x
    else:
        raise NotImplementedError("domain should be 'time' or 'freq'")

    if window is not None:
        if callable(window):
            W = window(cupy.fft.fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = cupy.fft.ifftshift(get_window(window, Nx))
        newshape = [1] * x.ndim
        newshape[axis] = len(W)
        W.shape = newshape
        X = X * cupy.asarray(W, dtype=X.dtype)

    sl = [slice(None)] * x.ndim
    newshape = list(x.shape)
    newshape[axis] = num
    N = int(cupy.minimum(num, Nx))
    nyq = N // 2 + 1  # Slice index that includes Nyquist
    Y = cupy.zeros(newshape, dtype=X.dtype)
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if N > 2:  # avoid empty slice
        sl[axis] = slice(nyq - N, None)
        Y[tuple(sl)] = X[tuple(sl)]

    # symmetrize nyquest freq bins if N is even
    if N % 2 == 0:
        if num < Nx:
            # select the component of Y at frequency +N/2,
            # add the component of X at -N/2
            sl[axis] = slice(-N // 2, -N // 2 + 1)
            Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N // 2, N // 2 + 1)
            Y[tuple(sl)] *= 0.5
            temp = Y[tuple(sl)]
            # set the component at -N/2 equal to the component at +N/2
            sl[axis] = slice(num - N // 2, num - N // 2 + 1)
            Y[tuple(sl)] = temp

    y = cupy.fft.ifft(Y, axis=axis) * (float(num) / float(Nx))

    if x.dtype.char not in ["F", "D"]:
        y = y.real

    if t is None:
        return y
    else:
        new_t = cupy.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t


def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0),
                  padtype='constant', cval=None):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. Values beyond the boundary of the signal are assumed to be zero
    during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by
        `cupyx.scipy.signal.upfirdn`. Changes assumptions on values beyond
        the boundary. If `constant`, assumed to be `cval` (default zero).
        If `line` assumed to continue a linear trend defined by the first and
        last points. `mean`, `median`, `maximum` and `minimum` work as in
        `cupy.pad` and assume that the values beyond the boundary are the mean,
        median, maximum or minimum respectively of the array along the axis.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `cusignal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `cusignal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `cusignal.get_window`
    and `cusignal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> import cupy
    >>> import cupyx.scipy.signal import resample, resample_poly

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f_fft = resample(y, 100)
    >>> f_poly = resample_poly(y, 100, 20)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cupy.asnumpy(xnew), cupy.asnumpy(f_fft), 'b.-', \
                 cupy.asnumpy(xnew), cupy.asnumpy(f_poly), 'r.-')
    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'ko-')
    >>> plt.plot(10, cupy.asnumpy(y[0]), 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()
    """

    x = cupy.asarray(x)
    if up != int(up):
        raise ValueError("up must be an integer")
    if down != int(down):
        raise ValueError("down must be an integer")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    if cval is not None and padtype != 'constant':
        raise ValueError('cval has no effect when padtype is ', padtype)

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_in = x.shape[axis]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list, cupy.ndarray)):
        window = cupy.array(window)  # use array to force a copy (we modify it)
        if window.ndim > 1:
            raise ValueError('window must be 1-D')
        half_len = (window.size - 1) // 2
        h = window
    else:
        # Design a linear-phase low-pass FIR filter
        max_rate = max(up, down)
        f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
        half_len = 10 * max_rate  # reasonable cutoff for sinc-like function
        h = firwin(2 * half_len + 1, f_c,
                   window=window).astype(x.dtype)  # match dtype of x
    h *= up

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = (down - half_len % down)
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in,
                      up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = cupy.concatenate((cupy.zeros(n_pre_pad, dtype=h.dtype), h,
                          cupy.zeros(n_post_pad, dtype=h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out

    # Remove background depending on the padtype option
    funcs = {'mean': cupy.mean, 'median': cupy.median,
             'minimum': cupy.amin, 'maximum': cupy.amax}
    upfirdn_kwargs = {'mode': 'constant', 'cval': 0}
    if padtype in funcs:
        background_values = funcs[padtype](x, axis=axis, keepdims=True)
    elif padtype in _upfirdn_modes:
        upfirdn_kwargs = {'mode': padtype}
        if padtype == 'constant':
            if cval is None:
                cval = 0
            upfirdn_kwargs['cval'] = cval
    else:
        raise ValueError(
            'padtype must be one of: maximum, mean, median, minimum, ' +
            ', '.join(_upfirdn_modes))

    if padtype in funcs:
        x = x - background_values

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis=axis, **upfirdn_kwargs)
    keep = [slice(None), ]*x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    y_keep = y[tuple(keep)]

    # Add background back
    if padtype in funcs:
        y_keep += background_values

    return y_keep

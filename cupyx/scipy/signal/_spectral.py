"""
Spectral analysis functions and utilities.

Some of the functions defined here were ported directly from CuSignal under
terms of the Apache license, under the following notice

Copyright (c) 2019-2020, NVIDIA CORPORATION.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import cupy
from cupyx.scipy.signal._spectral_impl import (
    _lombscargle, _spectral_helper, _median_bias)


def lombscargle(x, y, freqs, precenter=False, normalize=False):
    """
    lombscargle(x, y, freqs)

    Computes the Lomb-Scargle periodogram.

    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
    extended by Scargle [2]_ to find, and test the significance of weak
    periodic signals with uneven temporal sampling.

    When *normalize* is False (default) the computed periodogram
    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
    signal with amplitude A for sufficiently large N.

    When *normalize* is True the computed periodogram is normalized by
    the residuals of the data around a constant reference model (at zero).

    Input arrays should be one-dimensional and will be cast to float64.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values.
    freqs : array_like
        Angular frequencies for output periodogram.
    precenter : bool, optional
        Pre-center amplitudes by subtracting the mean.
    normalize : bool, optional
        Compute normalized periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    Notes
    -----
    This subroutine calculates the periodogram using a slightly
    modified algorithm due to Townsend [3]_ which allows the
    periodogram to be calculated using only a single pass through
    the input arrays for each frequency.
    The algorithm running time scales roughly as O(x * freqs) or O(N^2)
    for a large number of samples and frequencies.

    References
    ----------
    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976
    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -
           Statistical aspects of spectral analysis of unevenly spaced data",
           The Astrophysical Journal, vol 263, pp. 835-853, 1982
    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle
           periodogram using graphics processing units.", The Astrophysical
           Journal Supplement Series, vol 191, pp. 247-253, 2010

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    welch: Power spectral density by Welch's method
    spectrogram: Spectrogram by Welch's method
    csd: Cross spectral density by Welch's method
    """

    x = cupy.asarray(x, dtype=cupy.float64)
    y = cupy.asarray(y, dtype=cupy.float64)
    freqs = cupy.asarray(freqs, dtype=cupy.float64)
    pgram = cupy.empty(freqs.shape[0], dtype=cupy.float64)

    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    y_dot = cupy.zeros(1, dtype=cupy.float64)
    if normalize:
        cupy.dot(y, y, out=y_dot)

    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y

    _lombscargle(x, y_in, freqs, pgram, y_dot)

    return pgram


def periodogram(
    x,
    fs=1.0,
    window="boxcar",
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
):
    """
    Estimate power spectral density using a periodogram.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to 'boxcar'.
    nfft : int, optional
        Length of the FFT used. If `None` the length of `x` will be
        used.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of `x`.

    See Also
    --------
    welch: Estimate power spectral density using Welch's method
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    """
    x = cupy.asarray(x)

    if x.size == 0:
        return cupy.empty(x.shape), cupy.empty(x.shape)

    if window is None:
        window = "boxcar"

    if nfft is None:
        nperseg = x.shape[axis]
    elif nfft == x.shape[axis]:
        nperseg = nfft
    elif nfft > x.shape[axis]:
        nperseg = x.shape[axis]
    elif nfft < x.shape[axis]:
        # cp.s_ not implemented
        s = [cupy.s_[:]] * len(x.shape)
        s[axis] = cupy.s_[:nfft]
        x = cupy.asarray(x[tuple(s)])
        nperseg = nfft
        nfft = None

    return welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
    )


def welch(
    x,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    r"""
    Estimate power spectral density using Welch's method.

    Welch's method [1]_ computes an estimate of the power spectral
    density by dividing the data into overlapping segments, computing a
    modified periodogram for each segment and averaging the
    periodograms.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. Defaults to 'mean'.


    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of x.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    If `noverlap` is 0, this method is equivalent to Bartlett's method
    [2]_.

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """

    freqs, Pxx = csd(
        x,
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        average=average,
    )

    return freqs, Pxx.real


def csd(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's
    method.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    axis : int, optional
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. Defaults to 'mean'.


    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density or cross power spectrum of x,y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method. [Equivalent to
           csd(x,x)]
    coherence: Magnitude squared coherence by Welch's method.

    Notes
    -----
    By convention, Pxy is computed with the conjugate FFT of X
    multiplied by the FFT of Y.

    If the input series differ in length, the shorter series will be
    zero-padded to match.

    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    """
    x = cupy.asarray(x)
    y = cupy.asarray(y)
    freqs, _, Pxy = _spectral_helper(
        x,
        y,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling,
        axis,
        mode="psd",
    )

    # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if average == "median":
                Pxy = cupy.median(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])
            elif average == "mean":
                Pxy = Pxy.mean(axis=-1)
            else:
                raise ValueError(
                    'average must be "median" or "mean", got %s' % (average,)
                )
        else:
            Pxy = cupy.reshape(Pxy, Pxy.shape[:-1])

    return freqs, Pxy

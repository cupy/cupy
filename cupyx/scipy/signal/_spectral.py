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

import warnings

import cupy
from cupyx.scipy.signal.windows._windows import get_window
from cupyx.scipy.signal._spectral_impl import (
    _lombscargle, _spectral_helper, _median_bias, _triage_segments)


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
        # cupy.s_ not implemented
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


def check_COLA(window, nperseg, noverlap, tol=1e-10):
    r"""Check whether the Constant OverLap Add (COLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies COLA within `tol`,
        `False` otherwise

    See Also
    --------
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, it is sufficient that the signal windowing obeys the constraint of
    "Constant OverLap Add" (COLA). This ensures that every point in the input
    data is equally weighted, thereby avoiding aliasing and allowing full
    reconstruction.

    Some examples of windows that satisfy COLA:
        - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...
        - Bartlett window at overlap of 1/2, 3/4, 5/6, ...
        - Hann window at 1/2, 2/3, 3/4, ...
        - Any Blackman family window at 2/3 overlap
        - Any window with ``noverlap = nperseg-1``

    A very comprehensive list of other windows may be found in [2]_,
    wherein the COLA condition is satisfied when the "Amplitude
    Flatness" is unity. See [1]_ for more information.

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    """
    nperseg = int(nperseg)

    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    noverlap = int(noverlap)

    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    step = nperseg - noverlap
    binsums = sum(win[ii * step:(ii + 1) * step]
                  for ii in range(nperseg//step))

    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]

    deviation = binsums - cupy.median(binsums)
    return cupy.max(cupy.abs(deviation)) < tol


def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    r"""Check whether the Nonzero Overlap Add (NOLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies the NOLA constraint within
        `tol`, `False` otherwise

    See Also
    --------
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    for all :math:`n`, where :math:`w` is the window function, :math:`t` is the
    frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -
    `noverlap`).

    This ensures that the normalization factors in the denominator of the
    overlap-add inversion equation are not zero. Only very pathological windows
    will fail the NOLA constraint.

    See [1]_, [2]_ for more information.

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    """
    nperseg = int(nperseg)

    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg')
    if noverlap < 0:
        raise ValueError('noverlap must be a nonnegative integer')
    noverlap = int(noverlap)

    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    step = nperseg - noverlap
    binsums = sum(win[ii * step:(ii + 1) * step] ** 2
                  for ii in range(nperseg//step))

    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]**2

    return cupy.min(binsums) > tol


def stft(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend=False,
    return_onesided=True,
    boundary="zeros",
    padded=True,
    axis=-1,
    scaling='spectrum'
):
    r"""
    Compute the Short Time Fourier Transform (STFT).

    STFTs can be used as a way of quantifying the change of a
    nonstationary signal's frequency and phase content over time.

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
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`. When
        specified, the COLA constraint must be met (see Notes below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        Axis along which the STFT is computed; the default is over the
        last axis (i.e. ``axis=-1``).
    scaling: {'spectrum', 'psd'}
        The default 'spectrum' scaling allows each frequency line of `Zxx` to
        be interpreted as a magnitude spectrum. The 'psd' option scales each
        line to a power spectral density - it allows to calculate the signal's
        energy by numerically integrating over ``abs(Zxx)**2``.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of `x`. By default, the last axis of `Zxx` corresponds
        to the segment times.

    See Also
    --------
    welch: Power spectral density by Welch's method.
    spectrogram: Spectrogram by Welch's method.
    csd: Cross spectral density by Welch's method.
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "Nonzero
    OverLap Add" (NOLA), and the input signal must have complete
    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %
    (nperseg-noverlap) == 0``). The `padded` argument may be used to
    accomplish this.

    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop
    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index
    :math:`t` is given by

    .. math:: x_{t}[n]=x[n]w[n-tH]

    The overlap-add (OLA) reconstruction equation is given by

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    The NOLA constraint ensures that every normalization term that appears
    in the denomimator of the OLA reconstruction equation is nonzero. Whether a
    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can
    be tested with `check_NOLA`.

    See [1]_, [2]_ for more information.

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> import cupy
    >>> import cupyx.scipy.signal import stft
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * cupy.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = cupy.arange(N) / float(fs)
    >>> mod = 500*cupy.cos(2*cupy.pi*0.25*time)
    >>> carrier = amp * cupy.sin(2*cupy.pi*3e3*time + mod)
    >>> noise = cupy.random.normal(scale=cupy.sqrt(noise_power),
    ...                            size=time.shape)
    >>> noise *= cupy.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the STFT's magnitude.

    >>> f, t, Zxx = stft(x, fs, nperseg=1000)
    >>> plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(f),
    ...                cupy.asnumpy(cupy.abs(Zxx)), vmin=0, vmax=amp)
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    """
    if scaling == 'psd':
        scaling = 'density'
    elif scaling != 'spectrum':
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    freqs, time, Zxx = _spectral_helper(
        x,
        x,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling=scaling,
        axis=axis,
        mode="stft",
        boundary=boundary,
        padded=padded,
    )

    return freqs, time, Zxx


def istft(
    Zxx,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    input_onesided=True,
    boundary=True,
    time_axis=-1,
    freq_axis=-2,
    scaling='spectrum'
):
    r"""
    Perform the inverse Short Time Fourier transform (iSTFT).

    Parameters
    ----------
    Zxx : array_like
        STFT of the signal to be reconstructed. If a purely real array
        is passed, it will be cast to a complex data type.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window. Must match the window used to generate the
        STFT for faithful inversion.
    nperseg : int, optional
        Number of data points corresponding to each STFT segment. This
        parameter must be specified if the number of data points per
        segment is odd, or if the STFT was padded via ``nfft >
        nperseg``. If `None`, the value depends on the shape of
        `Zxx` and `input_onesided`. If `input_onesided` is `True`,
        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,
        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`, half
        of the segment length. Defaults to `None`. When specified, the
        COLA constraint must be met (see Notes below), and should match
        the parameter used to generate the STFT. Defaults to `None`.
    nfft : int, optional
        Number of FFT points corresponding to each STFT segment. This
        parameter must be specified if the STFT was padded via ``nfft >
        nperseg``. If `None`, the default values are the same as for
        `nperseg`, detailed above, with one exception: if
        `input_onesided` is True and
        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on
        that value. This case allows the proper inversion of an
        odd-length unpadded STFT using ``nfft=None``. Defaults to
        `None`.
    input_onesided : bool, optional
        If `True`, interpret the input array as one-sided FFTs, such
        as is returned by `stft` with ``return_onesided=True`` and
        `numpy.fft.rfft`. If `False`, interpret the input as a
        two-sided FFT. Defaults to `True`.
    boundary : bool, optional
        Specifies whether the input signal was extended at its
        boundaries by supplying a non-`None` ``boundary`` argument to
        `stft`. Defaults to `True`.
    time_axis : int, optional
        Where the time segments of the STFT is located; the default is
        the last axis (i.e. ``axis=-1``).
    freq_axis : int, optional
        Where the frequency axis of the STFT is located; the default is
        the penultimate axis (i.e. ``axis=-2``).
    scaling: {'spectrum', 'psd'}
        The default 'spectrum' scaling allows each frequency line of `Zxx` to
        be interpreted as a magnitude spectrum. The 'psd' option scales each
        line to a power spectral density - it allows to calculate the signal's
        energy by numerically integrating over ``abs(Zxx)**2``.

    Returns
    -------
    t : ndarray
        Array of output data times.
    x : ndarray
        iSTFT of `Zxx`.

    See Also
    --------
    stft: Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT with
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    This ensures that the normalization factors that appear in the denominator
    of the overlap-add reconstruction equation

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    are not zero. The NOLA constraint can be checked with the `check_NOLA`
    function.

    An STFT which has been modified (via masking or otherwise) is not
    guaranteed to correspond to a exactly realizible signal. This
    function implements the iSTFT via the least-squares estimation
    algorithm detailed in [2]_, which produces a signal that minimizes
    the mean squared error between the STFT of the returned signal and
    the modified STFT.

    See [1]_, [2]_ for more information.

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> import cupy
    >>> from cupyx.scipy.signal import stft, istft
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by
    0.001 V**2/Hz of white noise sampled at 1024 Hz.

    >>> fs = 1024
    >>> N = 10*fs
    >>> nperseg = 512
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.001 * fs / 2
    >>> time = cupy.arange(N) / float(fs)
    >>> carrier = amp * cupy.sin(2*cupy.pi*50*time)
    >>> noise = cupy.random.normal(scale=cupy.sqrt(noise_power),
    ...                          size=time.shape)
    >>> x = carrier + noise

    Compute the STFT, and plot its magnitude

    >>> f, t, Zxx = cusignal.stft(x, fs=fs, nperseg=nperseg)
    >>> f = cupy.asnumpy(f)
    >>> t = cupy.asnumpy(t)
    >>> Zxx = cupy.asnumpy(Zxx)
    >>> plt.figure()
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.ylim([f[1], f[-1]])
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.yscale('log')
    >>> plt.show()

    Zero the components that are 10% or less of the carrier magnitude,
    then convert back to a time series via inverse STFT

    >>> Zxx = cupy.where(cupy.abs(Zxx) >= amp/10, Zxx, 0)
    >>> _, xrec = cusignal.istft(Zxx, fs)
    >>> xrec = cupy.asnumpy(xrec)
    >>> x = cupy.asnumpy(x)
    >>> time = cupy.asnumpy(time)
    >>> carrier = cupy.asnumpy(carrier)

    Compare the cleaned signal with the original and true carrier signals.

    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([2, 2.1])*+
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    Note that the cleaned signal does not start as abruptly as the original,
    since some of the coefficients of the transient were also removed:

    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([0, 0.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()
    """

    # Make sure input is an ndarray of appropriate complex dtype
    Zxx = cupy.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    if Zxx.ndim < 2:
        raise ValueError("Input stft must be at least 2d!")

    if freq_axis == time_axis:
        raise ValueError("Must specify differing time and frequency axes!")

    nseg = Zxx.shape[time_axis]

    if input_onesided:
        # Assume even segment length
        n_default = 2 * (Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    # Check windowing parameters
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            # Odd nperseg, no FFT padding
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    nstep = nperseg - noverlap

    # Rearrange axes if necessary
    if time_axis != Zxx.ndim - 1 or freq_axis != Zxx.ndim - 2:
        # Turn negative indices to positive for the call to transpose
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        Zxx = cupy.transpose(Zxx, zouter + [freq_axis, time_axis])

    # Get window as array
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if win.shape[0] != nperseg:
            raise ValueError("window must have length of {0}".format(nperseg))

    ifunc = cupy.fft.irfft if input_onesided else cupy.fft.ifft
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # Initialize output and normalization arrays
    outputlength = nperseg + (nseg - 1) * nstep
    x = cupy.zeros(list(Zxx.shape[:-2]) + [outputlength], dtype=xsubs.dtype)
    norm = cupy.zeros(outputlength, dtype=xsubs.dtype)

    if cupy.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)

    if scaling == 'spectrum':
        xsubs *= win.sum()
    elif scaling == 'psd':
        xsubs *= cupy.sqrt(fs * cupy.sum(win**2))
    else:
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    for ii in range(nseg):
        # Window the ifft
        x[..., ii * nstep:ii * nstep + nperseg] += xsubs[..., ii] * win
        norm[..., ii * nstep:ii * nstep + nperseg] += win**2

    # Remove extension points
    if boundary:
        x = x[..., nperseg // 2: -(nperseg // 2)]
        norm = norm[..., nperseg // 2: -(nperseg // 2)]

    # Divide out normalization where non-tiny
    if cupy.sum(norm > 1e-10) != len(norm):
        warnings.warn("NOLA condition failed, STFT may not be invertible")
    x /= cupy.where(norm > 1e-10, norm, 1.0)

    if input_onesided:
        x = x.real

    # Put axes back
    if x.ndim > 1:
        if time_axis != Zxx.ndim - 1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = cupy.rollaxis(x, -1, time_axis)

    time = cupy.arange(x.shape[0]) / float(fs)
    return time, x


def spectrogram(
    x,
    fs=1.0,
    window=("tukey", 0.25),
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
):
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Spectrograms can be used as a way of visualizing the change of a
    nonstationary signal's frequency content over time.

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
        directly as the window and its length must be nperseg.
        Defaults to a Tukey window with shape parameter of 0.25.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 8``. Defaults to `None`.
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
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.
    axis : int, optional
        Axis along which the spectrogram is computed; the default is over
        the last axis (i.e. ``axis=-1``).
    mode : str, optional
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds
        to the segment times.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. In contrast to welch's method, where the
    entire data stream is averaged over, one may wish to use a smaller
    overlap (or perhaps none at all) when computing a spectrogram, to
    maintain some statistical independence between individual segments.
    It is for this reason that the default window is a Tukey window with
    1/8th of a window's length overlap at each end. See [1]_ for more
    information.

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.

    Examples
    --------
    >>> import cupy
    >>> from cupyx.scipy.signal import spectrogram
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * cupy.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = cupy.arange(N) / float(fs)
    >>> mod = 500*cupy.cos(2*cupy.pi*0.25*time)
    >>> carrier = amp * cupy.sin(2*cupy.pi*3e3*time + mod)
    >>> noise = cupy.random.normal(
    ...     scale=cupy.sqrt(noise_power), size=time.shape)
    >>> noise *= cupy.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the spectrogram.

    >>> f, t, Sxx = spectrogram(x, fs)
    >>> plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(f), cupy.asnumpy(Sxx))
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()

    Note, if using output that is not one sided, then use the following:

    >>> f, t, Sxx = spectrogram(x, fs, return_onesided=False)
    >>> plt.pcolormesh(cupy.asnumpy(t), cupy.fft.fftshift(f), \
        cupy.fft.fftshift(Sxx, axes=0))
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    """
    modelist = ["psd", "complex", "magnitude", "angle", "phase"]
    if mode not in modelist:
        raise ValueError(
            "unknown value for mode {}, must be one of {}".format(
                mode, modelist)
        )

    # need to set default for nperseg before setting default for noverlap below
    window, nperseg = _triage_segments(
        window, nperseg, input_length=x.shape[axis])

    # Less overlap than welch, so samples are more statisically independent
    if noverlap is None:
        noverlap = nperseg // 8

    if mode == "psd":
        freqs, time, Sxx = _spectral_helper(
            x,
            x,
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

    else:
        freqs, time, Sxx = _spectral_helper(
            x,
            x,
            fs,
            window,
            nperseg,
            noverlap,
            nfft,
            detrend,
            return_onesided,
            scaling,
            axis,
            mode="stft",
        )

        if mode == "magnitude":
            Sxx = cupy.abs(Sxx)
        elif mode in ["angle", "phase"]:
            Sxx = cupy.angle(Sxx)
            if mode == "phase":
                # Sxx has one additional dimension for time strides
                if axis < 0:
                    axis -= 1
                Sxx = cupy.unwrap(Sxx, axis=axis)

        # mode =='complex' is same as `stft`, doesn't need modification

    return freqs, time, Sxx


def coherence(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    axis=-1,
):
    r"""
    Estimate the magnitude squared coherence estimate, Cxy, of
    discrete-time signals X and Y using Welch's method.

    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power
    spectral density estimates of X and Y, and `Pxy` is the cross
    spectral density estimate of X and Y.

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
    axis : int, optional
        Axis along which the coherence is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Cxy : ndarray
        Magnitude squared coherence of x and y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap. See [1]_ and [2]_ for more
    information.

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
           Signals" Prentice Hall, 2005

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy.signal import butter, lfilter, coherence
    >>> import matplotlib.pyplot as plt

    Generate two test signals with some common features.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = cupy.arange(N) / fs
    >>> b, a = butter(2, 0.25, 'low')
    >>> x = cupy.random.normal(
    ...         scale=cupy.sqrt(noise_power), size=time.shape)
    >>> y = lfilter(b, a, x)
    >>> x += amp * cupy.sin(2*cupy.pi*freq*time)
    >>> y += cupy.random.normal(
    ...         scale=0.1*cupy.sqrt(noise_power), size=time.shape)

    Compute and plot the coherence.

    >>> f, Cxy = coherence(x, y, fs, nperseg=1024)
    >>> plt.semilogy(cupy.asnumpy(f), cupy.asnumpy(Cxy))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Coherence')
    >>> plt.show()
    """

    freqs, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        axis=axis,
    )
    _, Pyy = welch(
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        axis=axis,
    )
    _, Pxy = csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        axis=axis,
    )

    Cxy = cupy.abs(Pxy) ** 2 / Pxx / Pyy

    return freqs, Cxy


def vectorstrength(events, period):
    """
    Determine the vector strength of the events corresponding to the given
    period.

    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.

    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".

    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.

    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.

    Notes
    -----
    See [1]_, [2]_ and [3]_ for more information.

    References
    ----------
    .. [1] van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating
           vector strength: Auditory system, electric fish, and noise.
           Chaos 21, 047508 (2011).
    .. [2] van Hemmen, JL. Vector strength after Goldberg, Brown, and
           von Mises: biological and mathematical perspectives.  Biol Cybern.
           2013 Aug;107(4):385-96.
    .. [3] van Hemmen, JL and Vollmayr, AN.  Resonating vector strength:
           what happens when we vary the "probing" frequency while keeping
           the spike times fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
    """
    events = cupy.asarray(events)
    period = cupy.asarray(period)
    if events.ndim > 1:
        raise ValueError("events cannot have dimensions more than 1")
    if period.ndim > 1:
        raise ValueError("period cannot have dimensions more than 1")

    # we need to know later if period was originally a scalar
    scalarperiod = not period.ndim

    events = cupy.atleast_2d(events)
    period = cupy.atleast_2d(period)
    if (period <= 0).any():
        raise ValueError("periods must be positive")

    # this converts the times to vectors
    vectors = cupy.exp(cupy.dot(2j * cupy.pi / period.T, events))

    # the vector strength is just the magnitude of the mean of the vectors
    # the vector phase is the angle of the mean of the vectors
    vectormean = cupy.mean(vectors, axis=1)
    strength = cupy.abs(vectormean)
    phase = cupy.angle(vectormean)

    # if the original period was a scalar, return scalars
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
    return strength, phase

import operator
from math import pi

import cupy
from cupy.polynomial.polynomial import (
    polyval as npp_polyval, polyvalfromroots as npp_polyvalfromroots)
import cupyx.scipy.fft as sp_fft


def _try_convert_to_int(x):
    """Return an integer for ``5`` and ``array(5)``, fail if not an
       integer scalar.

    NB: would be easier if ``operator.index(cupy.array(5))`` worked
    (numpy.array(5) does)
    """
    if isinstance(x, cupy.ndarray):
        if x.ndim == 0:
            value = x.item()
        else:
            return x, False
    else:
        value = x
    try:
        return operator.index(value), True
    except TypeError:
        return value, False


def roots(arr):
    """np.roots replacement. XXX: calls into NumPy, then converts back.
    """
    import numpy as np

    arr = cupy.asarray(arr).get()
    return cupy.asarray(np.roots(arr))


def findfreqs(num, den, N, kind='ba'):
    """
    Find array of frequencies for computing the response of an analog filter.

    Parameters
    ----------
    num, den : array_like, 1-D
        The polynomial coefficients of the numerator and denominator of the
        transfer function of the filter or LTI system, where the coefficients
        are ordered from highest to lowest degree. Or, the roots  of the
        transfer function numerator and denominator (i.e., zeroes and poles).
    N : int
        The length of the array to be computed.
    kind : str {'ba', 'zp'}, optional
        Specifies whether the numerator and denominator are specified by their
        polynomial coefficients ('ba'), or their roots ('zp').

    Returns
    -------
    w : (N,) ndarray
        A 1-D array of frequencies, logarithmically spaced.

    See Also
    --------
    scipy.signal.find_freqs

    Examples
    --------
    Find a set of nine frequencies that span the "interesting part" of the
    frequency response for the filter with the transfer function

        H(s) = s / (s^2 + 8s + 25)

    >>> from scipy import signal
    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)
    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,
             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,
             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])
    """
    if kind == 'ba':
        ep = cupy.atleast_1d(roots(den)) + 0j
        tz = cupy.atleast_1d(roots(num)) + 0j
    elif kind == 'zp':
        ep = cupy.atleast_1d(den) + 0j
        tz = cupy.atleast_1d(num) + 0j
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")

    if len(ep) == 0:
        ep = cupy.atleast_1d(-1000) + 0j

    ez = cupy.r_[
                 cupy.compress(ep.imag >= 0, ep, axis=-1),
                 cupy.compress((abs(tz) < 1e5) & (tz.imag >= 0), tz, axis=-1)]

    integ = cupy.abs(ez) < 1e-10
    hfreq = cupy.around(cupy.log10(cupy.max(3 * cupy.abs(ez.real + integ) +
                                            1.5 * ez.imag)) + 0.5)
    lfreq = cupy.around(cupy.log10(0.1 * cupy.min(cupy.abs((ez + integ).real) +
                                                  2 * ez.imag)) - 0.5)
    w = cupy.logspace(lfreq, hfreq, N)
    return w


def freqs(b, a, worN=200, plot=None):
    """
    Compute frequency response of analog filter.

    Given the M-order numerator `b` and N-order denominator `a` of an analog
    filter, compute its frequency response::

             b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
     H(w) = ----------------------------------------------
             a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    scipy.signal.freqs
    freqz : Compute the frequency response of a digital filter.

    """
    if worN is None:
        # For backwards compatibility
        w = findfreqs(b, a, 200)

    else:
        N, _is_int = _try_convert_to_int(worN)
        if _is_int:
            w = findfreqs(b, a, N)
        else:
            w = cupy.atleast_1d(worN)

    s = 1j * w
    h = cupy.polyval(b, s) / cupy.polyval(a, s)

    if plot is not None:
        plot(w, h)

    return w, h


def freqs_zpk(z, p, k, worN=200):
    """
    Compute frequency response of analog filter.

    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its
    frequency response::

                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])
     H(w) = k * ----------------------------------------
                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqz : Compute the frequency response of a digital filter in TF form
    freqz_zpk : Compute the frequency response of a digital filter in ZPK form

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import freqs_zpk, iirfilter

    >>> z, p, k = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1',
    ...                     output='zpk')

    >>> w, h = freqs_zpk(z, p, k, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid(True)
    >>> plt.show()

    """
    k = np.asarray(k)
    if k.size > 1:
        raise ValueError('k must be a single scalar gain')

    if worN is None:
        # For backwards compatibility
        w = findfreqs(z, p, 200, kind='zp')
    elif _is_int_type(worN):
        w = findfreqs(z, p, worN, kind='zp')
    else:
        w = worN

    w = atleast_1d(w)
    s = 1j * w
    num = polyvalfromroots(s, z)
    den = polyvalfromroots(s, p)
    h = k * num/den
    return w, h


def freqz(b, a=1, worN=512, whole=False, plot=None, fs=2*pi,
          include_nyquist=False):
    """
    Compute the frequency response of a digital filter.

    Given the M-order numerator `b` and N-order denominator `a` of a digital
    filter, compute its frequency response::

                 jw                 -jw              -jwM
        jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
     H(e  ) = ------ = -----------------------------------
                 jw                 -jw              -jwN
              A(e  )    a[0] + a[1]e    + ... + a[N]e

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    a : array_like
        Denominator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512). This is a convenient alternative to::

            cupy.linspace(0, fs if whole else fs/2, N,
                          endpoint=include_nyquist)

        Using a number that is fast for FFT computations can result in
        faster computations (see Notes).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if worN is array_like.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqz`.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).
    include_nyquist : bool, optional
        If `whole` is False and `worN` is an integer, setting `include_nyquist`
        to True will include the last frequency (Nyquist frequency) and is
        otherwise ignored.

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz_zpk
    sosfreqz
    scipy.signal.freqz


    Notes
    -----
    Using Matplotlib's :func:`matplotlib.pyplot.plot` function as the callable
    for `plot` produces unexpected results, as this plots the real part of the
    complex transfer function, not the magnitude.
    Try ``lambda w, h: plot(w, cupy.abs(h))``.

    A direct computation via (R)FFT is used to compute the frequency response
    when the following conditions are met:

    1. An integer value is given for `worN`.
    2. `worN` is fast to compute via FFT (i.e.,
       `next_fast_len(worN) <scipy.fft.next_fast_len>` equals `worN`).
    3. The denominator coefficients are a single value (``a.shape[0] == 1``).
    4. `worN` is at least as long as the numerator coefficients
       (``worN >= b.shape[0]``).
    5. If ``b.ndim > 1``, then ``b.shape[-1] == 1``.

    For long FIR filters, the FFT approach can have lower error and be much
    faster than the equivalent direct polynomial calculation.
    """
    b = cupy.atleast_1d(b)
    a = cupy.atleast_1d(a)

    if worN is None:
        # For backwards compatibility
        worN = 512

    h = None

    N, _is_int = _try_convert_to_int(worN)
    if _is_int:
        if N < 0:
            raise ValueError(f'worN must be nonnegative, got {N}')
        lastpoint = 2 * pi if whole else pi

        # if include_nyquist is true and whole is false, w should
        # include end point
        w = cupy.linspace(
            0, lastpoint, N, endpoint=include_nyquist and not whole)

        use_fft = (a.size == 1 and
                   N >= b.shape[0] and
                   sp_fft.next_fast_len(N) == N and
                   (b.ndim == 1 or (b.shape[-1] == 1))
                   )

        if use_fft:
            # if N is fast, 2 * N will be fast, too, so no need to check
            n_fft = N if whole else N * 2
            if cupy.isrealobj(b) and cupy.isrealobj(a):
                fft_func = sp_fft.rfft
            else:
                fft_func = sp_fft.fft

            h = fft_func(b, n=n_fft, axis=0)[:N]
            h /= a
            if fft_func is sp_fft.rfft and whole:
                # exclude DC and maybe Nyquist (no need to use axis_reverse
                # here because we can build reversal with the truncation)
                stop = -1 if n_fft % 2 == 1 else -2
                h_flip = slice(stop, 0, -1)
                h = cupy.concatenate((h, h[h_flip].conj()))
            if b.ndim > 1:
                # Last axis of h has length 1, so drop it.
                h = h[..., 0]
                # Move the first axis of h to the end.
                h = cupy.moveaxis(h, 0, -1)
    else:
        w = cupy.atleast_1d(worN)
        w = 2 * pi * w / fs

    if h is None:  # still need to compute using freqs w
        zm1 = cupy.exp(-1j * w)
        h = (npp_polyval(zm1, b, tensor=False) /
             npp_polyval(zm1, a, tensor=False))

    w = w * fs / (2 * pi)

    if plot is not None:
        plot(w, h)

    return w, h


def freqz_zpk(z, p, k, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in ZPK form.

    Given the Zeros, Poles and Gain of a digital filter, compute its frequency
    response:

    :math:`H(z)=k \prod_i (z - Z[i]) / \prod_j (z - P[j])`

    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are
    the `poles`.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqs_zpk : Compute the frequency response of an analog filter in ZPK form
    freqz : Compute the frequency response of a digital filter in TF form
    scipy.signal.freqz_zpk

    """
    z, p = map(cupy.atleast_1d, (z, p))

    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi

    if worN is None:
        # For backwards compatibility
        w = cupy.linspace(0, lastpoint, 512, endpoint=False)
    else:
        N, _is_int = _try_convert_to_int(worN)
        if _is_int:
            w = cupy.linspace(0, lastpoint, N, endpoint=False)
        else:
            w = cupy.atleast_1d(worN)
            w = 2 * pi * w / fs

    zm1 = cupy.exp(1j * w)
    h = k * npp_polyvalfromroots(zm1, z) / npp_polyvalfromroots(zm1, p)

    w = w * fs / (2 * pi)

    return w, h


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = cupy.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


def sosfreqz(sos, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in SOS format.

    Given `sos`, an array with shape (n, 6) of second order sections of
    a digital filter, compute the frequency response of the system function::

               B0(z)   B1(z)         B{n-1}(z)
        H(z) = ----- * ----- * ... * ---------
               A0(z)   A1(z)         A{n-1}(z)

    for z = exp(omega*1j), where B{k}(z) and A{k}(z) are numerator and
    denominator of the transfer function of the k-th second order section.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).  Using a number that is fast for FFT computations can result
        in faster computations (see Notes of `freqz`).

        If an array_like, compute the response at the frequencies given (must
        be 1-D). These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz, sosfilt
    scipy.signal.sosfreqz
    """
    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    h = 1.
    for row in sos:
        w, rowh = freqz(row[:3], row[3:], worN=worN, whole=whole, fs=fs)
        h *= rowh
    return w, h

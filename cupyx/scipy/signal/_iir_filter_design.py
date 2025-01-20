"""IIR filter design APIs"""
from math import pi
import math

import cupy

from cupyx.scipy.signal._iir_filter_conversions import (
    lp2bp_zpk, lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear_zpk, zpk2tf, zpk2sos)
from cupyx.scipy.signal._iir_filter_conversions import (
    buttap, cheb1ap, cheb2ap, ellipap, buttord, ellipord, cheb1ord, cheb2ord,
    _validate_gpass_gstop)


# FIXME

def besselap():
    raise NotImplementedError


bessel_norms = {'fix': 'me'}


def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter
    coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).

        When Wn is a length-2 sequence, ``Wn[0]`` must be less than ``Wn[1]``.
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec
    scipy.signal.iirfilter

    """
    ftype, btype, output = [x.lower() for x in (ftype, btype, output)]

    Wn = cupy.asarray(Wn)
    # if cupy.any(Wn <= 0):
    #    raise ValueError("filter critical frequencies must be greater than 0")

    if Wn.size > 1 and not Wn[0] < Wn[1]:
        raise ValueError("Wn[0] must be less than Wn[1]")

    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = 2*Wn/fs

    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError(
            "'%s' is an invalid bandtype for filter." % btype) from e

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError as e:
        raise ValueError(
            "'%s' is not a valid basic IIR filter." % ftype) from e

    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)

    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    elif typefunc == besselap:
        z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    elif typefunc == ellipap:
        if rs is None or rp is None:
            raise ValueError("Both rp and rs must be provided to design an "
                             "elliptic filter.")
        z, p, k = typefunc(N, rp, rs)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # Pre-warp frequencies for digital filter design
    if not analog:
        if cupy.any(Wn <= 0) or cupy.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies must "
                                 f"be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs/2})")
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * cupy.tan(pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if cupy.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn '
                             'for lowpass or highpass filter')

        if btype == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = cupy.sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError('Wn must specify start and stop frequencies for '
                             'bandpass or bandstop filter') from e

        if btype == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k, analog=analog)


def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter. For 'bandpass' and 'bandstop' filters,
        the resulting order of the final second-order sections ('sos')
        matrix is ``2*N``, with `N` the number of biquad sections
        of the desired system.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.

        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").

        For digital filters, if `fs` is not specified, `Wn` units are
        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is
        thus in half cycles / sample and defined as 2*critical frequencies
        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.

        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    buttord, buttap
    iirfilter
    scipy.signal.butter


    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.

    If the transfer function form ``[b, a]`` is requested, numerical
    problems can occur since the conversion between roots and
    the polynomial coefficients is a numerically sensitive operation,
    even for N >= 4. It is recommended to work with the SOS
    representation.

    .. warning::
        Designing high-order and narrowband IIR filters in TF form can
        result in unstable or incorrect filtering due to floating point
        numerical precision issues. Consider inspecting output filter
        characteristics `freqz` or designing the filters with second-order
        sections via ``output='sos'``.
    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)


def cheby1(N, rp, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type I digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type I filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type I filters, this is the point in the transition band at which
        the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    cheb1ord, cheb1ap
    iirfilter
    scipy.signal.cheby1

    Notes
    -----
    The Chebyshev type I filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the passband and increased ringing in the step response.

    Type I filters roll off faster than Type II (`cheby2`), but Type II
    filters do not have any ripple in the passband.

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.
    """
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='cheby1', fs=fs)


def cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type II digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type II filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type II filters, this is the point in the transition band at which
        the gain first reaches -`rs`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    cheb2ord, cheb2ap
    iirfilter
    scipy.signal.cheby2

    Notes
    -----
    The Chebyshev type II filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the stopband and increased ringing in the step response.

    Type II filters do not roll off as fast as Type I (`cheby1`).
    """
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog,
                     output=output, ftype='cheby2', fs=fs)


def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Elliptic (Cauer) digital and analog filter design.

    Design an Nth-order digital or analog elliptic filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For elliptic filters, this is the point in the transition band at
        which the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter. Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    ellipord, ellipap
    iirfilter
    scipy.signal.ellip

    Notes
    -----
    Also known as Cauer or Zolotarev filters, the elliptical filter maximizes
    the rate of transition between the frequency response's passband and
    stopband, at the expense of ripple in both, and increased ringing in the
    step response.

    As `rp` approaches 0, the elliptical filter becomes a Chebyshev
    type II filter (`cheby2`). As `rs` approaches 0, it becomes a Chebyshev
    type I filter (`cheby1`). As both approach 0, it becomes a Butterworth
    filter (`butter`).

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.
    """
    return iirfilter(N, Wn, rs=rs, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='elliptic', fs=fs)


def iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba',
              fs=None):
    """Complete IIR digital and analog filter design.

    Given passband and stopband frequencies and gains, construct an analog or
    digital IIR filter of minimum order for a given basic type. Return the
    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order
    sections ('sos') form.

    Parameters
    ----------
    wp, ws : float or array like, shape (2,)
        Passband and stopband edge frequencies. Possible values are scalars
        (for lowpass and highpass filters) or ranges (for bandpass and bandstop
        filters).
        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s). Note, that for bandpass and bandstop filters passband
        must lie strictly inside stopband or vice versa.
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    scipy.signal.iirdesign
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    """
    try:
        ordfunc = filter_dict[ftype][1]
    except KeyError as e:
        raise ValueError("Invalid IIR filter type: %s" % ftype) from e
    except IndexError as e:
        raise ValueError(("%s does not have order selection. Use "
                          "iirfilter function.") % ftype) from e

    _validate_gpass_gstop(gpass, gstop)

    wp = cupy.atleast_1d(wp)
    ws = cupy.atleast_1d(ws)

    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError("wp and ws must have one or two elements each, and"
                         "the same shape, got %s and %s"
                         % (wp.shape, ws.shape))

    if any(wp <= 0) or any(ws <= 0):
        raise ValueError("Values for wp, ws must be greater than 0")

    if not analog:
        if fs is None:
            if any(wp >= 1) or any(ws >= 1):
                raise ValueError("Values for wp, ws must be less than 1")
        elif any(wp >= fs/2) or any(ws >= fs/2):
            raise ValueError("Values for wp, ws must be less than fs/2"
                             " (fs={} -> fs/2={})".format(fs, fs/2))

    if wp.shape[0] == 2:
        if not ((ws[0] < wp[0] and wp[1] < ws[1]) or
                (wp[0] < ws[0] and ws[1] < wp[1])):
            raise ValueError("Passband must lie strictly inside stopband"
                             " or vice versa")

    band_type = 2 * (len(wp) - 1)
    band_type += 1
    if wp[0] >= ws[0]:
        band_type += 1

    btype = {1: 'lowpass', 2: 'highpass',
             3: 'bandstop', 4: 'bandpass'}[band_type]

    N, Wn = ordfunc(wp, ws, gpass, gstop, analog=analog, fs=fs)
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype,
                     ftype=ftype, output=output, fs=fs)


def iircomb(w0, Q, ftype='notch', fs=2.0, *, pass_zero=False):
    """
    Design IIR notching or peaking digital comb filter.

    A notching comb filter consists of regularly-spaced band-stop filters with
    a narrow bandwidth (high quality factor). Each rejects a narrow frequency
    band and leaves the rest of the spectrum little changed.

    A peaking comb filter consists of regularly-spaced band-pass filters with
    a narrow bandwidth (high quality factor). Each rejects components outside
    a narrow frequency band.

    Parameters
    ----------
    w0 : float
        The fundamental frequency of the comb filter (the spacing between its
        peaks). This must evenly divide the sampling frequency. If `fs` is
        specified, this is in the same units as `fs`. By default, it is
        a normalized scalar that must satisfy  ``0 < w0 < 1``, with
        ``w0 = 1`` corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : {'notch', 'peak'}
        The type of comb filter generated by the function. If 'notch', then
        the Q factor applies to the notches. If 'peak', then the Q factor
        applies to the peaks.  Default is 'notch'.
    fs : float, optional
        The sampling frequency of the signal. Default is 2.0.
    pass_zero : bool, optional
        If False (default), the notches (nulls) of the filter are centered on
        frequencies [0, w0, 2*w0, ...], and the peaks are centered on the
        midpoints [w0/2, 3*w0/2, 5*w0/2, ...].  If True, the peaks are centered
        on [0, w0, 2*w0, ...] (passing zero frequency) and vice versa.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    Raises
    ------
    ValueError
        If `w0` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `fs` is not divisible by `w0`, if `ftype`
        is not 'notch' or 'peak'

    See Also
    --------
    scipy.signal.iircomb
    iirnotch
    iirpeak

    Notes
    -----
    The TF implementation of the
    comb filter is numerically stable even at higher orders due to the
    use of a single repeated pole, which won't suffer from precision loss.

    References
    ----------
    Sophocles J. Orfanidis, "Introduction To Signal Processing",
         Prentice-Hall, 1996, ch. 11, "Digital Filter Design"
    """

    # Convert w0, Q, and fs to float
    w0 = float(w0)
    Q = float(Q)
    fs = float(fs)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    if not 0 < w0 < fs / 2:
        raise ValueError("w0 must be between 0 and {}"
                         " (nyquist), but given {}.".format(fs / 2, w0))
    if ftype not in ('notch', 'peak'):
        raise ValueError('ftype must be either notch or peak.')

    # Compute the order of the filter
    N = round(fs / w0)

    # Check for cutoff frequency divisibility
    if abs(w0 - fs/N)/fs > 1e-14:
        raise ValueError('fs must be divisible by w0.')

    # Compute frequency in radians and filter bandwidth
    # Eq. 11.3.1 (p. 574) from reference [1]
    w0 = (2 * pi * w0) / fs
    w_delta = w0 / Q

    # Define base gain values depending on notch or peak filter
    # Compute -3dB attenuation
    # Eqs. 11.4.1 and 11.4.2 (p. 582) from reference [1]
    if ftype == 'notch':
        G0, G = 1, 0
    elif ftype == 'peak':
        G0, G = 0, 1
    GB = 1 / math.sqrt(2)

    # Compute beta
    # Eq. 11.5.3 (p. 591) from reference [1]
    beta = math.sqrt((GB**2 - G0**2) / (G**2 - GB**2)) * \
        math.tan(N * w_delta / 4)

    # Compute filter coefficients
    # Eq 11.5.1 (p. 590) variables a, b, c from reference [1]
    ax = (1 - beta) / (1 + beta)
    bx = (G0 + G * beta) / (1 + beta)
    cx = (G0 - G * beta) / (1 + beta)

    # Last coefficients are negative to get peaking comb that passes zero or
    # notching comb that doesn't.
    negative_coef = ((ftype == 'peak' and pass_zero) or
                     (ftype == 'notch' and not pass_zero))

    # Compute numerator coefficients
    # Eq 11.5.1 (p. 590) or Eq 11.5.4 (p. 591) from reference [1]
    # b - cz^-N or b + cz^-N
    b = cupy.zeros(N + 1)
    b[0] = bx
    if negative_coef:
        b[-1] = -cx
    else:
        b[-1] = +cx

    # Compute denominator coefficients
    # Eq 11.5.1 (p. 590) or Eq 11.5.4 (p. 591) from reference [1]
    # 1 - az^-N or 1 + az^-N
    a = cupy.zeros(N + 1)
    a[0] = 1
    if negative_coef:
        a[-1] = -ax
    else:
        a[-1] = +ax

    return b, a


def iirnotch(w0, Q, fs=2.0):
    """
    Design second-order IIR notch digital filter.

    A notch filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    Parameters
    ----------
    w0 : float
        Frequency to remove from a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    scipy.signal.iirnotch

    References
    ----------
    Sophocles J. Orfanidis, "Introduction To Signal Processing",
         Prentice-Hall, 1996
    """

    return _design_notch_peak_filter(w0, Q, "notch", fs)


def iirpeak(w0, Q, fs=2.0):
    """
    Design second-order IIR peak (resonant) digital filter.

    A peak filter is a band-pass filter with a narrow bandwidth
    (high quality factor). It rejects components outside a narrow
    frequency band.

    Parameters
    ----------
    w0 : float
        Frequency to be retained in a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        peak filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.


    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    scpy.signal.iirpeak

    References
    ----------
    Sophocles J. Orfanidis, "Introduction To Signal Processing",
       Prentice-Hall, 1996
    """

    return _design_notch_peak_filter(w0, Q, "peak", fs)


def _design_notch_peak_filter(w0, Q, ftype, fs=2.0):
    """
    Design notch or peak digital filter.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. If `fs` is specified,
        this is in the same units as `fs`. By default, it is a normalized
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:

            - notch filter : ``notch``
            - peak filter  : ``peak``
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """

    # Guarantee that the inputs are floats
    w0 = float(w0)
    Q = float(Q)
    w0 = 2 * w0 / fs

    # Checks if w0 is within the range
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError("w0 should be such that 0 < w0 < 1")

    # Get bandwidth
    bw = w0 / Q

    # Normalize inputs
    bw = bw * pi
    w0 = w0 * pi

    # Compute -3dB attenuation
    gb = 1 / math.sqrt(2)

    if ftype == "notch":
        # Compute beta: formula 11.3.4 (p.575) from reference [1]
        beta = (math.sqrt(1.0 - gb**2.0) / gb) * math.tan(bw / 2.0)
    elif ftype == "peak":
        # Compute beta: formula 11.3.19 (p.579) from reference [1]
        beta = (gb / math.sqrt(1.0 - gb**2.0)) * math.tan(bw / 2.0)
    else:
        raise ValueError("Unknown ftype.")

    # Compute gain: formula 11.3.6 (p.575) from reference [1]
    gain = 1.0 / (1.0 + beta)

    # Compute numerator b and denominator a
    # formulas 11.3.7 (p.575) and 11.3.21 (p.579)
    # from reference [1]
    if ftype == "notch":
        b = [gain * x for x in (1.0, -2.0 * math.cos(w0), 1.0)]
    else:
        b = [(1.0 - gain) * x for x in (1.0, 0.0, -1.0)]

    a = [1.0, -2.0 * gain * math.cos(w0), 2.0 * gain - 1.0]

    a = cupy.asarray(a)
    b = cupy.asarray(b)

    return b, a


filter_dict = {'butter': [buttap, buttord],
               'butterworth': [buttap, buttord],

               'cauer': [ellipap, ellipord],
               'elliptic': [ellipap, ellipord],
               'ellip': [ellipap, ellipord],

               'bessel': [besselap],
               'bessel_phase': [besselap],
               'bessel_delay': [besselap],
               'bessel_mag': [besselap],

               'cheby1': [cheb1ap, cheb1ord],
               'chebyshev1': [cheb1ap, cheb1ord],
               'chebyshevi': [cheb1ap, cheb1ord],

               'cheby2': [cheb2ap, cheb2ord],
               'chebyshev2': [cheb2ap, cheb2ord],
               'chebyshevii': [cheb2ap, cheb2ord],
               }

band_dict = {'band': 'bandpass',
             'bandpass': 'bandpass',
             'pass': 'bandpass',
             'bp': 'bandpass',

             'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',

             'l': 'lowpass',
             'low': 'lowpass',
             'lowpass': 'lowpass',
             'lp': 'lowpass',

             'high': 'highpass',
             'highpass': 'highpass',
             'h': 'highpass',
             'hp': 'highpass',
             }

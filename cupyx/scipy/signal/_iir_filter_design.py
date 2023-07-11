"""IIR filter design APIs"""
from math import pi

import cupy

from cupyx.scipy.signal import (
    lp2bp_zpk, lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear_zpk, zpk2tf, zpk2sos)
from cupyx.scipy.signal._iir_filter_conversions import (
    buttap, cheb1ap, cheb2ap, ellipap)


# FIXME

def besselap():
    raise NotImplementedError


def buttord():
    raise NotImplementedError


def ellipord():
    raise NotImplementedError


def cheb1ord():
    raise NotImplementedError


def cheb2ord():
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

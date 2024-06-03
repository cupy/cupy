"""Functions for FIR filter design."""
import math

from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window

import cupy
import numpy


__all__ = ["firls", "minimum_phase"]


def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.

    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.

    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.

    See Also
    --------
    scipy.signal.kaiser_beta

    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta


def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.

    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:

        a = 2.285 * (N - 1) * pi * width + 7.95

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter,
        expressed as a fraction of the Nyquist frequency.

    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.

    See Also
    --------
    scipy.signal.kaiser_atten
    """
    a = 2.285 * (numtaps - 1) * cupy.pi * width + 7.95
    return a


def kaiserord(ripple, width):
    """
    Determine the filter window parameters for the Kaiser window method.

    The parameters returned by this function are generally used to create
    a finite impulse response filter using the window method, with either
    `firwin` or `firwin2`.

    Parameters
    ----------
    ripple : float
        Upper bound for the deviation (in dB) of the magnitude of the
        filter's frequency response from that of the desired filter (not
        including frequencies in any transition intervals). That is, if w
        is the frequency expressed as a fraction of the Nyquist frequency,
        A(w) is the actual frequency response of the filter and D(w) is the
        desired frequency response, the design requirement is that::

            abs(A(w) - D(w))) < 10**(-ripple/20)

        for 0 <= w <= 1 and w not in a transition interval.
    width : float
        Width of transition region, normalized so that 1 corresponds to pi
        radians / sample. That is, the frequency is expressed as a fraction
        of the Nyquist frequency.

    Returns
    -------
    numtaps : int
        The length of the Kaiser window.
    beta : float
        The beta parameter for the Kaiser window.

    See Also
    --------
    scipy.signal.kaiserord


    """
    A = abs(ripple)  # in case somebody is confused as to what's meant
    if A < 8:
        # Formula for N is not valid in this range.
        raise ValueError("Requested maximum ripple attenuation %f is too "
                         "small for the Kaiser formula." % A)
    beta = kaiser_beta(A)

    # Kaiser's formula (as given in Oppenheim and Schafer) is for the filter
    # order, so we have to add 1 to get the number of taps.
    numtaps = (A - 7.95) / 2.285 / (cupy.pi * width) + 1

    return int(numpy.ceil(numtaps)), beta


_firwin_kernel = cupy.ElementwiseKernel(
    "float64 win, int32 numtaps, raw float64 bands, int32 steps, bool scale",
    "float64 h, float64 hc",
    """
    const double m { static_cast<double>( i ) - alpha ?
        static_cast<double>( i ) - alpha : 1.0e-20 };

    double temp {};
    double left {};
    double right {};

    for ( int s = 0; s < steps; s++ ) {
        left = bands[s * 2 + 0] ? bands[s * 2 + 0] : 1.0e-20;
        right = bands[s * 2 + 1] ? bands[s * 2 + 1] : 1.0e-20;

        temp += right * ( sin( right * m * M_PI ) / ( right * m * M_PI ) );
        temp -= left * ( sin( left * m * M_PI ) / ( left * m * M_PI ) );
    }

    temp *= win;
    h = temp;

    double scale_frequency {};

    if ( scale ) {
        left = bands[0];
        right = bands[1];

        if ( left == 0 ) {
            scale_frequency = 0.0;
        } else if ( right == 1 ) {
            scale_frequency = 1.0;
        } else {
            scale_frequency = 0.5 * ( left + right );
        }
        double c { cos( M_PI * m * scale_frequency ) };
        hc = temp * c;
    }
    """,
    "_firwin_kernel",
    options=("-std=c++11",),
    loop_prep="const double alpha { 0.5 * ( numtaps - 1 ) };",
)


# Scipy <= 1.12 has a deprecated `nyq` argument (nyq = fs/2).
# Remove it here, to be forward-looking.
def firwin(
    numtaps,
    cutoff,
    width=None,
    window="hamming",
    pass_zero=True,
    scale=True,
    fs=2,
):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`.  The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `cusignal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'},
        optional
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise
    fs : float, optional
        The sampling frequency of the signal.  Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to ``fs/2``, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See Also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> import cusignal
    >>> numtaps = 3
    >>> f = 0.1
    >>> cusignal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> cusignal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> cusignal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> cusignal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> cusignal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """

    nyq = 0.5 * fs

    cutoff = cupy.atleast_1d(cutoff) / float(nyq)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError(
            "The cutoff argument must be at most " "one-dimensional.")
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError(
            "Invalid cutoff frequency: frequencies must be "
            "greater than 0 and less than nyq."
        )
    if cupy.any(cupy.diff(cutoff) <= 0):
        raise ValueError(
            "Invalid cutoff frequencies: the frequencies "
            "must be strictly increasing."
        )

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ("kaiser", beta)

    if isinstance(pass_zero, str):
        if pass_zero in ("bandstop", "lowpass"):
            if pass_zero == "lowpass":
                if cutoff.size != 1:
                    raise ValueError(
                        "cutoff must have one element if "
                        'pass_zero=="lowpass", got %s' % (cutoff.shape,)
                    )
            elif cutoff.size <= 1:
                raise ValueError(
                    "cutoff must have at least two elements if "
                    'pass_zero=="bandstop", got %s' % (cutoff.shape,)
                )
            pass_zero = True
        elif pass_zero in ("bandpass", "highpass"):
            if pass_zero == "highpass":
                if cutoff.size != 1:
                    raise ValueError(
                        "cutoff must have one element if "
                        'pass_zero=="highpass", got %s' % (cutoff.shape,)
                    )
            elif cutoff.size <= 1:
                raise ValueError(
                    "cutoff must have at least two elements if "
                    'pass_zero=="bandpass", got %s' % (cutoff.shape,)
                )
            pass_zero = False
        else:
            raise ValueError(
                'pass_zero must be True, False, "bandpass", '
                '"lowpass", "highpass", or "bandstop", got '
                "{}".format(pass_zero)
            )

    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero

    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError(
            "A filter with an even number of coefficients must "
            "have zero response at the Nyquist rate."
        )

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = cupy.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    win = get_window(window, numtaps, fftbins=False)
    h, hc = _firwin_kernel(win, numtaps, bands, bands.shape[0], scale)
    if scale:
        s = cupy.sum(hc)
        h /= s

        # Build up the coefficients.
        alpha = 0.5 * (numtaps - 1)
        m = cupy.arange(0, numtaps) - alpha
        h = 0
        for left, right in bands:
            h += right * cupy.sinc(right * m)
            h -= left * cupy.sinc(left * m)

        h *= win

        # Now handle scaling if desired.
        if scale:
            # Get the first passband.
            left, right = bands[0]
            if left == 0:
                scale_frequency = 0.0
            elif right == 1:
                scale_frequency = 1.0
            else:
                scale_frequency = 0.5 * (left + right)
            c = cupy.cos(cupy.pi * m * scale_frequency)
            s = cupy.sum(h * c)
            h /= s

    return h


def firwin2(
    numtaps,
    freq,
    gain,
    nfreqs=None,
    window="hamming",
    nyq=None,
    antisymmetric=False,
    fs=2.0,
):
    """
    FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``. Default is 2.

    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.

    See Also
    --------
    scipy.signal.firwin2
    firls
    firwin
    minimum_phase
    remez

    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.
    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:

       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced

    Magnitude response of all but type I filters are subjects to following
    constraints:

       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency
    """
    nyq = 0.5 * fs

    if len(freq) != len(gain):
        raise ValueError("freq and gain must be of same length.")

    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(
            (
                "ntaps must be less than nfreqs, but firwin2 was "
                "called with ntaps=%d and nfreqs=%s"
            )
            % (numtaps, nfreqs)
        )

    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError("freq must start with 0 and end with fs/2.")
    d = cupy.diff(freq)
    if (d < 0).any():
        raise ValueError("The values in freq must be nondecreasing.")
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError("A value in freq must not occur more than twice.")
    if freq[1] == 0:
        raise ValueError("Value 0 must not be repeated in freq")
    if freq[-2] == nyq:
        raise ValueError("Value fs/2 must not be repeated in freq")

    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    else:
        if numtaps % 2 == 0:
            ftype = 2
        else:
            ftype = 1

    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError(
            "A Type II filter must have zero gain at the " "Nyquist frequency."
        )
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError(
            "A Type III filter must have zero gain at zero "
            "and Nyquist frequencies."
        )
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError(
            "A Type IV filter must have zero gain at zero " "frequency.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(math.ceil(math.log(numtaps, 2)))

    if (d == 0).any():
        # Tweak any repeated values in freq so that interp works.
        freq = cupy.array(freq, copy=True)
        eps = cupy.finfo(float).eps * nyq
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # Check if freq is strictly increasing after tweak
        d = cupy.diff(freq)
        if (d <= 0).any():
            raise ValueError(
                "freq cannot contain numbers that are too close "
                "(within eps * (fs/2): "
                "{}) to a repeated value".format(eps)
            )

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = cupy.linspace(0.0, nyq, nfreqs)
    fx = cupy.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = cupy.exp(-(numtaps - 1) / 2.0 * 1.0j * math.pi * x / nyq)
    if ftype > 2:
        shift *= 1j

    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = cupy.fft.irfft(fx2)

    if window is not None:
        # Create the window to apply to the filter coefficients.
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind

    if ftype == 3:
        out[out.size // 2] = 0.0

    return out


# Scipy <= 1.12 has a deprecated `nyq` argument (nyq = fs/2).
# Remove it here, to be forward-looking.
def firls(numtaps, bands, desired, weight=None, fs=2):
    """
    FIR filter design using least-squares error minimization.

    Calculate the filter coefficients for the linear-phase finite
    impulse response (FIR) filter which has the best approximation
    to the desired frequency response described by `bands` and
    `desired` in the least squares sense (i.e., the integral of the
    weighted mean-squared error within the specified bands is
    minimized).

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter. `numtaps` must be odd.
    bands : array_like
        A monotonic nondecreasing sequence containing the band edges in
        Hz. All elements must be non-negative and less than or equal to
        the Nyquist frequency given by `fs`/2. The bands are specified as
        frequency pairs, thus, if using a 1D array, its length must be
        even, e.g., `cupy.array([0, 1, 2, 3, 4, 5])`. Alternatively, the
        bands can be specified as an nx2 sized 2D array, where n is the
        number of bands, e.g, `cupy.array([[0, 1], [2, 3], [4, 5]])`.
        All elements of `bands` must be monotonically nondecreasing, have
        width > 0, and must not overlap. (This is not checked by the routine).
    desired : array_like
        A sequence the same size as `bands` containing the desired gain
        at the start and end point of each band.
        All elements must be non-negative (this is not checked by the routine).
    weight : array_like, optional
        A relative weighting to give to each band region when solving
        the least squares problem. `weight` has to be half the size of
        `bands`.
        All elements must be non-negative (this is not checked by the routine).
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `bands`
        must be between 0 and ``fs/2`` (inclusive). Default is 2.

    Returns
    -------
    coeffs : ndarray
        Coefficients of the optimal (in a least squares sense) FIR filter.

    See Also
    --------
    firwin
    firwin2
    minimum_phase
    remez
    scipy.signal.firls
    """
    nyq = 0.5 * fs

    numtaps = int(numtaps)
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError("numtaps must be odd and >= 1")
    M = (numtaps-1) // 2

    # normalize bands 0->1 and make it 2 columns
    nyq = float(nyq)
    if nyq <= 0:
        raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
    bands = cupy.asarray(bands).flatten() / nyq
    if len(bands) % 2 != 0:
        raise ValueError("bands must contain frequency pairs.")
    if (bands < 0).any() or (bands > 1).any():
        raise ValueError("bands must be between 0 and 1 relative to Nyquist")
    bands.shape = (-1, 2)

    # check remaining params
    desired = cupy.asarray(desired).flatten()
    if bands.size != desired.size:
        raise ValueError("desired must have one entry per frequency, got %s "
                         "gains for %s frequencies."
                         % (desired.size, bands.size))
    desired.shape = (-1, 2)
    # if (cupy.diff(bands) <= 0).any() or (cupy.diff(bands[:, 0]) < 0).any():
    #     raise ValueError("bands must be monotonically nondecreasing and have"
    #                     " width > 0.")
    # if (bands[:-1, 1] > bands[1:, 0]).any():
    #     raise ValueError("bands must not overlap.")
    # if (desired < 0).any():
    #     raise ValueError("desired must be non-negative.")
    if weight is None:
        weight = cupy.ones(len(desired))
    weight = cupy.asarray(weight).flatten()
    if len(weight) != len(desired):
        raise ValueError("weight must be the same size as the number of "
                         "band pairs ({}).".format(len(bands)))
    # if (weight < 0).any():
    #    raise ValueError("weight must be non-negative.")

    # Set up the linear matrix equation to be solved, Qa = b

    # We can express Q(k,n) = 0.5 Q1(k,n) + 0.5 Q2(k,n)
    # where Q1(k,n)=q(k-n) and Q2(k,n)=q(k+n), i.e. a Toeplitz plus Hankel.

    # We omit the factor of 0.5 above, instead adding it during coefficient
    # calculation.

    # We also omit the 1/π from both Q and b equations, as they cancel
    # during solving.

    # We have that:
    #     q(n) = 1/π ∫W(ω)cos(nω)dω (over 0->π)
    # Using our normalization ω=πf and with a constant weight W over each
    # interval f1->f2 we get:
    #     q(n) = W∫cos(πnf)df (0->1) = Wf sin(πnf)/πnf
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = cupy.arange(numtaps)[:, cupy.newaxis, cupy.newaxis]
    q = cupy.dot(cupy.diff(cupy.sinc(bands * n) *
                           bands, axis=2)[:, :, 0], weight)

    # Now we assemble our sum of Toeplitz and Hankel
    Q1 = toeplitz(q[:M+1])
    Q2 = hankel(q[:M+1], q[M:])
    Q = Q1 + Q2

    # Now for b(n) we have that:
    #     b(n) = 1/π ∫ W(ω)D(ω)cos(nω)dω (over 0->π)
    # Using our normalization ω=πf and with a constant weight W over each
    # interval and a linear term for D(ω) we get (over each f1->f2 interval):
    #     b(n) = W ∫ (mf+c)cos(πnf)df
    #          = f(mf+c)sin(πnf)/πnf + mf**2 cos(nπf)/(πnf)**2
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = n[:M + 1]  # only need this many coefficients here
    # Choose m and c such that we are at the start and end weights
    m = (cupy.diff(desired, axis=1) / cupy.diff(bands, axis=1))
    c = desired[:, [0]] - bands[:, [0]] * m
    b = bands * (m*bands + c) * cupy.sinc(bands * n)
    # Use L'Hospital's rule here for cos(nπf)/(πnf)**2 @ n=0
    b[0] -= m * bands * bands / 2.
    b[1:] += m * cupy.cos(n[1:] * cupy.pi * bands) / (cupy.pi * n[1:]) ** 2
    b = cupy.diff(b, axis=2)[:, :, 0] @ weight

    # Now we can solve the equation : XXX CuPy failure modes (?)
    with cupyx.errstate(linalg="raise"):
        try:
            a = solve(Q, b)
        except LinAlgError:
            # in case Q is rank deficient
            a = lstsq(Q, b, rcond=None)[0]

    # XXX: scipy.signal does this:
    # try:  # try the fast way
    #     with warnings.catch_warnings(record=True) as w:
    #         warnings.simplefilter('always')
    #         a = solve(Q, b)
    #     for ww in w:
    #         if (ww.category == LinAlgWarning and
    #                 str(ww.message).startswith('Ill-conditioned matrix')):
    #             raise LinAlgError(str(ww.message))
    # except LinAlgError:  # in case Q is rank deficient
    #     a = lstsq(Q, b)[0]

    # make coefficients symmetric (linear phase)
    coeffs = cupy.hstack((a[:0:-1], 2 * a[0], a[1:]))
    return coeffs


def _dhtm(mag):
    """Compute the modified 1-D discrete Hilbert transform

    Parameters
    ----------
    mag : ndarray
        The magnitude spectrum. Should be 1-D with an even length, and
        preferably a fast length for FFT/IFFT.
    """
    # Adapted based on code by Niranjan Damera-Venkata,
    # Brian L. Evans and Shawn R. McCaslin (see refs for `minimum_phase`)
    sig = cupy.zeros(len(mag))
    # Leave Nyquist and DC at 0, knowing np.abs(fftfreq(N)[midpt]) == 0.5
    midpt = len(mag) // 2
    sig[1:midpt] = 1
    sig[midpt + 1:] = -1
    # eventually if we want to support complex filters, we will need a
    # cupy.abs() on the mag inside the log, and should remove the .real
    recon = ifft(mag * cupy.exp(fft(sig * ifft(cupy.log(mag))))).real
    return recon


def minimum_phase(h, method='homomorphic', n_fft=None):
    """Convert a linear-phase FIR filter to minimum phase

    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.
    method : {'hilbert', 'homomorphic'}
        The method to use:

            'homomorphic' (default)
                This method [4]_ [5]_ works best with filters with an
                odd number of taps, and the resulting minimum phase filter
                will have a magnitude response that approximates the square
                root of the original filter's magnitude response.

            'hilbert'
                This method [1]_ is designed to be used with equiripple
                filters (e.g., from `remez`) with unity or zero gain
                regions.

    n_fft : int
        The number of points to use for the FFT. Should be at least a
        few times larger than the signal length (see Notes).

    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(length(h) + 1) // 2``.

    See Also
    --------
    scipy.signal.minimum_phase

    Notes
    -----
    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection
    of an FFT length to estimate the complex cepstrum of the filter.

    In the case of the Hilbert method, the deviation from the ideal
    spectrum ``epsilon`` is related to the number of stopband zeros
    ``n_stop`` and FFT length ``n_fft`` as::

        epsilon = 2. * n_stop / n_fft

    For example, with 100 stopband zeros and a FFT length of 2048,
    ``epsilon = 0.0976``. If we conservatively assume that the number of
    stopband zeros is one less than the filter length, we can take the FFT
    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::

        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))

    This gives reasonable results for both the Hilbert and homomorphic
    methods, and gives the value used when ``n_fft=None``.

    Alternative implementations exist for creating minimum-phase filters,
    including zero inversion [2]_ and spectral factorization [3]_ [4]_ [5]_.
    For more information, see:

        http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters

    References
    ----------
    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and
           complex minimum phase digital FIR filters," Acoustics, Speech,
           and Signal Processing, 1999. Proceedings., 1999 IEEE International
           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.
           DOI:10.1109/ICASSP.1999.756179
    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR
           filters by direct factorization," Signal Processing,
           vol. 10, no. 4, pp. 369-383, Jun. 1986.
    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in
           Handbook for Digital Signal Processing, chapter 4,
           New York: Wiley-Interscience, 1993.
    .. [4] J. S. Lim, Advanced Topics in Signal Processing.
           Englewood Cliffs, N.J.: Prentice Hall, 1988.
    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,
           "Discrete-Time Signal Processing," 2nd edition.
           Upper Saddle River, N.J.: Prentice Hall, 1999.

    """  # noqa
    if cupy.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1-D and at least 2 samples long')
    n_half = len(h) // 2
    if not cupy.allclose(h[-n_half:][::-1], h[:n_half]):
        import warnings
        warnings.warn('h does not appear to by symmetric, conversion may '
                      'fail', RuntimeWarning)
    if not isinstance(method, str) or method not in \
            ('homomorphic', 'hilbert',):
        raise ValueError('method must be "homomorphic" or "hilbert", got %r'
                         % (method,))
    if n_fft is None:
        n_fft = 2 ** int(cupy.ceil(cupy.log2(2 * (len(h) - 1) / 0.01)))
    n_fft = int(n_fft)
    if n_fft < len(h):
        raise ValueError('n_fft must be at least len(h)==%s' % len(h))
    if method == 'hilbert':
        w = cupy.arange(n_fft) * (2 * cupy.pi / n_fft * n_half)
        H = cupy.real(fft(h, n_fft) * cupy.exp(1j * w))
        dp = max(H) - 1
        ds = 0 - min(H)
        S = 4. / (cupy.sqrt(1 + dp + ds) + cupy.sqrt(1 - dp + ds)) ** 2
        H += ds
        H *= S
        H = cupy.sqrt(H, out=H)
        H += 1e-10  # ensure that the log does not explode
        h_minimum = _dhtm(H)
    else:  # method == 'homomorphic'
        # zero-pad; calculate the DFT
        h_temp = cupy.abs(fft(h, n_fft))
        # take 0.25*log(|H|**2) = 0.5*log(|H|)
        h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
        cupy.log(h_temp, out=h_temp)
        h_temp *= 0.5
        # IDFT
        h_temp = ifft(h_temp).real
        # multiply pointwise by the homomorphic filter
        # lmin[n] = 2u[n] - d[n]
        win = cupy.zeros(n_fft)
        win[0] = 1
        stop = (len(h) + 1) // 2
        win[1:stop] = 2
        if len(h) % 2:
            win[stop] = 1
        h_temp *= win
        h_temp = ifft(cupy.exp(fft(h_temp)))
        h_minimum = h_temp.real
    n_out = n_half + len(h) % 2
    return h_minimum[:n_out]

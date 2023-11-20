""" IIR filter conversion utilities.

Split off _filter_design.py
"""
import warnings
import math
from math import pi, prod

import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize

from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize


class BadCoefficients(UserWarning):
    """Warning about badly conditioned filter coefficients"""
    pass


def _trim_zeros(filt, trim='fb'):
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L1800-L1850

    first = 0
    if 'f' in trim:
        for i in filt:
            if i != 0.:
                break
            else:
                first = first + 1

    last = len(filt)
    if 'b' in trim:
        for i in filt[::-1]:
            if i != 0.:
                break
            else:
                last = last - 1
    return filt[first:last]


def _align_nums(nums):
    """Aligns the shapes of multiple numerators.

    Given an array of numerator coefficient arrays [[a_1, a_2,...,
    a_n],..., [b_1, b_2,..., b_m]], this function pads shorter numerator
    arrays with zero's so that all numerators have the same length. Such
    alignment is necessary for functions like 'tf2ss', which needs the
    alignment when dealing with SIMO transfer functions.

    Parameters
    ----------
    nums: array_like
        Numerator or list of numerators. Not necessarily with same length.

    Returns
    -------
    nums: array
        The numerator. If `nums` input was a list of numerators then a 2-D
        array with padded zeros for shorter numerators is returned. Otherwise
        returns ``np.asarray(nums)``.
    """
    try:
        # The statement can throw a ValueError if one
        # of the numerators is a single digit and another
        # is array-like e.g. if nums = [5, [1, 2, 3]]
        nums = cupy.asarray(nums)
        return nums

    except ValueError:
        nums = [cupy.atleast_1d(num) for num in nums]
        max_width = max(num.size for num in nums)

        # pre-allocate
        aligned_nums = cupy.zeros((len(nums), max_width))

        # Create numerators with padded zeros
        for index, num in enumerate(nums):
            aligned_nums[index, -num.size:] = num

        return aligned_nums


def _polycoeffs_from_zeros(zeros, tol=10):
    # a clone of numpy.poly, simplified
    dtyp = (cupy.complex_
            if cupy.issubdtype(zeros.dtype, cupy.complexfloating)
            else cupy.float_)
    a = cupy.ones(1, dtype=dtyp)
    for z in zeros:
        a = cupy.convolve(a, cupy.r_[1, -z], mode='full')

    # Use real output if possible.
    if dtyp == cupy.complex_:
        mask = cupy.abs(a.imag) < tol * cupy.finfo(a.dtype).eps
        a.imag[mask] = 0.0
        if mask.shape[0] == a.shape[0]:
            # all imag parts were fp noise
            a = a.real.copy()
        else:
            # if all cmplx roots are complex conj, the coefficients are real
            pos_roots = z[z.imag > 0]
            neg_roots = z[z.imag < 0]
            if pos_roots.shape[0] == neg_roots.shape[0]:
                neg_roots = neg_roots.copy()
                neg_roots.sort()
                pos_roots = pos_roots.copy()
                pos_roots.sort()
                if (neg_roots == pos_roots.conj()).all():
                    a = a.real.copy()
    return a


def _nearest_real_complex_idx(fro, to, which):
    """Get the next closest real or complex element based on distance"""
    assert which in ('real', 'complex', 'any')
    order = cupy.argsort(cupy.abs(fro - to))
    if which == 'any':
        return order[0]
    else:
        mask = cupy.isreal(fro[order])
        if which == 'complex':
            mask = ~mask
        return order[cupy.nonzero(mask)[0][0]]


def _single_zpksos(z, p, k):
    """Create one second-order section from up to two zeros and poles"""
    sos = cupy.zeros(6)
    b, a = zpk2tf(cupy.asarray(z), cupy.asarray(p), k)
    sos[3-len(b):3] = b
    sos[6-len(a):6] = a
    return sos


def zpk2sos(z, p, k, pairing=None, *, analog=False):
    """Return second-order sections from zeros, poles, and gain of a system

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        If analog is False and pairing is None, pairing is set to 'nearest';
        if analog is True, pairing must be 'minimal', and is set to that if
        it is None.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    sosfilt
    scipy.signal.zpk2sos

    """
    if pairing is None:
        pairing = 'minimal' if analog else 'nearest'

    valid_pairings = ['nearest', 'keep_odd', 'minimal']
    if pairing not in valid_pairings:
        raise ValueError('pairing must be one of %s, not %s'
                         % (valid_pairings, pairing))

    if analog and pairing != 'minimal':
        raise ValueError('for analog zpk2sos conversion, '
                         'pairing must be "minimal"')

    if len(z) == len(p) == 0:
        if not analog:
            return cupy.array([[k, 0., 0., 1., 0., 0.]])
        else:
            return cupy.array([[0., 0., k, 0., 0., 1.]])

    if pairing != 'minimal':
        # ensure we have the same number of poles and zeros, and make copies
        p = cupy.concatenate((p, cupy.zeros(max(len(z) - len(p), 0))))
        z = cupy.concatenate((z, cupy.zeros(max(len(p) - len(z), 0))))
        n_sections = (max(len(p), len(z)) + 1) // 2

        if len(p) % 2 == 1 and pairing == 'nearest':
            p = cupy.concatenate((p, cupy.zeros(1)))
            z = cupy.concatenate((z, cupy.zeros(1)))
        assert len(p) == len(z)
    else:
        if len(p) < len(z):
            raise ValueError('for analog zpk2sos conversion, '
                             'must have len(p)>=len(z)')

        n_sections = (len(p) + 1) // 2

    # Ensure we have complex conjugate pairs
    # (note that _cplxreal only gives us one element of each complex pair):
    z = cupy.concatenate(_cplxreal(z))
    p = cupy.concatenate(_cplxreal(p))
    if not cupy.isreal(k):
        raise ValueError('k must be real')
    k = k.real

    if not analog:
        # digital: "worst" is the closest to the unit circle
        def idx_worst(p):
            return cupy.argmin(cupy.abs(1 - cupy.abs(p)))
    else:
        # analog: "worst" is the closest to the imaginary axis
        def idx_worst(p):
            return cupy.argmin(cupy.abs(cupy.real(p)))

    sos = cupy.zeros((n_sections, 6))

    # Construct the system, reversing order so the "worst" are last
    for si in range(n_sections-1, -1, -1):
        # Select the next "worst" pole
        p1_idx = idx_worst(p)
        p1 = p[p1_idx]
        p = cupy.delete(p, p1_idx)

        # Pair that pole with a zero

        if cupy.isreal(p1) and cupy.isreal(p).sum() == 0:
            # Special case (1): last remaining real pole
            if pairing != 'minimal':
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = cupy.delete(z, z1_idx)
                sos[si] = _single_zpksos(cupy.r_[z1, 0], cupy.r_[p1, 0], 1)
            elif len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = cupy.delete(z, z1_idx)
                sos[si] = _single_zpksos([z1], [p1], 1)
            else:
                sos[si] = _single_zpksos([], [p1], 1)

        elif (len(p) + 1 == len(z)
              and not cupy.isreal(p1)
              and cupy.isreal(p).sum() == 1
              and cupy.isreal(z).sum() == 1):

            # Special case (2): there's one real pole and one real zero
            # left, and an equal number of poles and zeros to pair up.
            # We *must* pair with a complex zero

            z1_idx = _nearest_real_complex_idx(z, p1, 'complex')
            z1 = z[z1_idx]
            z = cupy.delete(z, z1_idx)
            sos[si] = _single_zpksos(
                cupy.r_[z1, z1.conj()], cupy.r_[p1, p1.conj()], 1)

        else:
            if cupy.isreal(p1):
                prealidx = cupy.flatnonzero(cupy.isreal(p))
                p2_idx = prealidx[idx_worst(p[prealidx])]
                p2 = p[p2_idx]
                p = cupy.delete(p, p2_idx)
            else:
                p2 = p1.conj()

            # find closest zero
            if len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'any')
                z1 = z[z1_idx]
                z = cupy.delete(z, z1_idx)

                if not cupy.isreal(z1):
                    sos[si] = _single_zpksos(
                        cupy.r_[z1, z1.conj()], cupy.r_[p1, p2], 1)
                else:
                    if len(z) > 0:
                        z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                        z2 = z[z2_idx]
                        assert cupy.isreal(z2)
                        z = cupy.delete(z, z2_idx)
                        sos[si] = _single_zpksos(cupy.r_[z1, z2], [p1, p2], 1)
                    else:
                        sos[si] = _single_zpksos([z1], [p1, p2], 1)
            else:
                # no more zeros
                sos[si] = _single_zpksos([], [p1, p2], 1)

    assert len(p) == len(z) == 0  # we've consumed all poles and zeros
    del p, z

    # put gain in first sos
    sos[0][:3] *= k
    return sos


def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.

    The 1-D input vector `z` is split up into its complex (zc) and real (zr)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.

    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    scipy.signal.cmplxreal

    Examples
    --------
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """

    z = cupy.atleast_1d(z)
    if z.size == 0:
        return z, z
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')

    if tol is None:
        # Get tolerance from dtype of input
        tol = 100 * cupy.finfo((1.0 * z).dtype).eps

    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    z = z[cupy.lexsort(cupy.array([abs(z.imag), z.real]))]

    # Split reals from conjugate pairs
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # Input is entirely real
        return cupy.array([]), zr

    # Split positive and negative halves of conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    same_real = cupy.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = cupy.diff(cupy.r_[0, same_real, 0])
    run_starts = cupy.nonzero(diffs > 0)[0]
    run_stops = cupy.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[cupy.lexsort(cupy.array([abs(chunk.imag)]))]

    # Check that negatives match positives
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    zc = (zp + zn.conj()) / 2

    return zc, zr


def normalize(b, a):
    """Normalize numerator/denominator of a continuous-time transfer function.

    If values of `b` are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    Parameters
    ----------
    b: array_like
        Numerator of the transfer function. Can be a 2-D array to normalize
        multiple transfer functions.
    a: array_like
        Denominator of the transfer function. At most 1-D.

    Returns
    -------
    num: array
        The numerator of the normalized transfer function. At least a 1-D
        array. A 2-D array if the input `num` is a 2-D array.
    den: 1-D array
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).

    See Also
    --------
    scipy.signal.normalize

    """
    num, den = b, a

    den = cupy.atleast_1d(den)
    num = cupy.atleast_2d(_align_nums(num))

    if den.ndim != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if num.ndim > 2:
        raise ValueError("Numerator polynomial must be rank-1 or"
                         " rank-2 array.")
    if cupy.all(den == 0):
        raise ValueError("Denominator must have at least on nonzero element.")

    # Trim leading zeros in denominator, leave at least one.
    den = _trim_zeros(den, 'f')

    # Normalize transfer function
    num, den = num / den[0], den / den[0]

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if cupy.allclose(col, 0, atol=1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        warnings.warn("Badly conditioned filter coefficients (numerator): the "
                      "results may be meaningless", BadCoefficients)
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num[0, :]

    return num, den


def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk
    bilinear
    scipy.signal.bilinear_zpk

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0 * fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = cupy.append(z_z, -cupy.ones(degree))

    # Compensate for gain change
    k_z = k * (cupy.prod(fs2 - z) / cupy.prod(fs2 - p)).real

    return z_z, p_z, k_z


def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.

    See Also
    --------
    lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2lp
    scipy.signal.lp2lp_zpk

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp


def lp2hp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp
    scipy.signal.lp2hp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = cupy.append(z_hp, cupy.zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * cupy.real(cupy.prod(-z) / cupy.prod(-p))

    return z_hp, p_hp, k_hp


def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear
    lp2bp
    scipy.signal.lp2bp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = cupy.concatenate((z_lp + cupy.sqrt(z_lp**2 - wo**2),
                             z_lp - cupy.sqrt(z_lp**2 - wo**2)))
    p_bp = cupy.concatenate((p_lp + cupy.sqrt(p_lp**2 - wo**2),
                             p_lp - cupy.sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = cupy.append(z_bp, cupy.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp


def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of the transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, bilinear
    lp2bs
    scipy.signal.lp2bs_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = cupy.concatenate((z_hp + cupy.sqrt(z_hp**2 - wo**2),
                             z_hp - cupy.sqrt(z_hp**2 - wo**2)))
    p_bs = cupy.concatenate((p_hp + cupy.sqrt(p_hp**2 - wo**2),
                             p_hp - cupy.sqrt(p_hp**2 - wo**2)))

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = cupy.append(z_bs, cupy.full(degree, +1j*wo))
    z_bs = cupy.append(z_bs, cupy.full(degree, -1j*wo))

    # Cancel out gain change caused by inversion
    k_bs = k * cupy.real(cupy.prod(-z) / cupy.prod(-p))

    return z_bs, p_bs, k_bs


def bilinear(b, a, fs=1.0):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    b : array_like
        Numerator of the analog filter transfer function.
    a : array_like
        Denominator of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    b : ndarray
        Numerator of the transformed digital filter transfer function.
    a : ndarray
        Denominator of the transformed digital filter transfer function.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, lp2bs
    bilinear_zpk
    scipy.signal.bilinear

    """
    fs = float(fs)
    a, b = map(cupy.atleast_1d, (a, b))
    D = a.shape[0] - 1
    N = b.shape[0] - 1

    M = max(N, D)
    Np, Dp = M, M

    bprime = cupy.empty(Np + 1, float)
    aprime = cupy.empty(Dp + 1, float)

    # XXX (ev-br): worth turning into a ufunc invocation? (loops are short)
    for j in range(Dp + 1):
        val = 0.0
        for i in range(N + 1):
            bNi = b[N - i] * (2 * fs)**i
            for k in range(i + 1):
                for s in range(M - i + 1):
                    if k + s == j:
                        val += comb(i, k) * comb(M - i, s) * bNi * (-1)**k
        bprime[j] = cupy.real(val)

    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            aDi = a[D - i] * (2 * fs)**i
            for k in range(i + 1):
                for s in range(M - i + 1):
                    if k + s == j:
                        val += comb(i, k) * comb(M - i, s) * aDi * (-1)**k
        aprime[j] = cupy.real(val)

    return normalize(bprime, aprime)


def lp2lp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g. rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed low-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed low-pass filter.

    See Also
    --------
    lp2hp, lp2bp, lp2bs, bilinear
    lp2lp_zpk
    scipy.signal.lp2lp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    """
    a, b = map(cupy.atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    M = max(d, n)
    pwo = wo ** cupy.arange(M - 1, -1, -1)
    start1 = max((n - d, 0))
    start2 = max((d - n, 0))
    b = b * pwo[start1] / pwo[start2:]
    a = a * pwo[start1] / pwo[start1:]
    return normalize(b, a)


def lp2hp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed high-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed high-pass
        filter.

    See Also
    --------
    lp2lp, lp2bp, lp2bs, bilinear
    lp2hp_zpk
    scipy.signal.lp2hp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.
    """
    a, b = map(cupy.atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    if wo != 1:
        pwo = wo ** cupy.arange(max(d, n))
    else:
        pwo = cupy.ones(max(d, n), b.dtype)
    if d >= n:
        outa = a[::-1] * pwo
        outb = cupy.resize(b, (d,))
        outb[n:] = 0.0
        outb[:n] = b[::-1] * pwo[:n]
    else:
        outb = b[::-1] * pwo
        outa = cupy.resize(a, (n,))
        outa[d:] = 0.0
        outa[:d] = a[::-1] * pwo[:d]

    return normalize(outb, outa)


def lp2bp(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-pass
        filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bs, bilinear
    lp2bp_zpk
    scipy.signal.lp2bp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    """
    a, b = map(cupy.atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = cupy.mintypecode((a.dtype, b.dtype))
    ma = max(N, D)
    Np = N + ma
    Dp = D + ma
    bprime = cupy.empty(Np + 1, artype)
    aprime = cupy.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * b[N - i] * (wosq) ** (i - k) / bw ** i
        bprime[Np - j] = val

    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * a[D - i] * (wosq) ** (i - k) / bw ** i
        aprime[Dp - j] = val

    return normalize(bprime, aprime)


def lp2bs(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-stop filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-stop
        filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, bilinear
    lp2bs_zpk
    scipy.signal.lp2bs

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.
    """
    a, b = map(cupy.atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = cupy.mintypecode((a.dtype, b.dtype))
    M = max(N, D)
    Np = M + M
    Dp = M + M
    bprime = cupy.empty(Np + 1, artype)
    aprime = cupy.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * b[N - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        bprime[Np - j] = val

    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * a[D - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        aprime[Dp - j] = val

    return normalize(bprime, aprime)


# ### LTI conversions ###

def zpk2tf(z, p, k):
    """
    Return polynomial transfer function representation from zeros and poles

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    scipy.signal.zpk2tf
    """
    if z.ndim > 1:
        raise NotImplementedError(f"zpk2tf: z.ndim = {z.ndim}.")
    b = _polycoeffs_from_zeros(z) * k
    a = _polycoeffs_from_zeros(p)
    return b, a


def tf2zpk(b, a):
    r"""Return zero, pole, gain (z, p, k) representation from a numerator,
    denominator representation of a linear filter.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.tf2zpk

    Notes
    -----
    If some values of `b` are too close to 0, they are removed. In that case,
    a BadCoefficients warning is emitted.

    The `b` and `a` arrays are interpreted as coefficients for positive,
    descending powers of the transfer function variable. So the inputs
    :math:`b = [b_0, b_1, ..., b_M]` and :math:`a =[a_0, a_1, ..., a_N]`
    can represent an analog filter of the form:

    .. math::

        H(s) = \frac
        {b_0 s^M + b_1 s^{(M-1)} + \cdots + b_M}
        {a_0 s^N + a_1 s^{(N-1)} + \cdots + a_N}

    or a discrete-time filter of the form:

    .. math::

        H(z) = \frac
        {b_0 z^M + b_1 z^{(M-1)} + \cdots + b_M}
        {a_0 z^N + a_1 z^{(N-1)} + \cdots + a_N}

    This "positive powers" form is found more commonly in controls
    engineering. If `M` and `N` are equal (which is true for all filters
    generated by the bilinear transform), then this happens to be equivalent
    to the "negative powers" discrete-time form preferred in DSP:

    .. math::

        H(z) = \frac
        {b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}
        {a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}}

    Although this is true for common filters, remember that this is not true
    in the general case. If `M` and `N` are not equal, the discrete-time
    transfer function coefficients must first be converted to the "positive
    powers" form before finding the poles and zeros.

    """
    b, a = normalize(b, a)
    b = (b + 0.0) / a[0]
    a = (a + 0.0) / a[0]
    k = b[0].copy()
    b /= b[0]
    z = roots(b)
    p = roots(a)
    return z, p, k


def tf2sos(b, a, pairing=None, *, analog=False):
    """
    Return second-order sections from transfer function representation

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See `zpk2sos` for information and restrictions on `pairing` and
        `analog` arguments.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    scipy.signal.tf2sos

    Notes
    -----
    It is generally discouraged to convert from TF to SOS format, since doing
    so usually will not improve numerical precision errors. Instead, consider
    designing filters in ZPK format and converting directly to SOS. TF is
    converted to SOS by first converting to ZPK format, then converting
    ZPK to SOS.

    """
    return zpk2sos(*tf2zpk(b, a), pairing=pairing, analog=analog)


def sos2tf(sos):
    """
    Return a single transfer function from a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    scipy.signal.sos2tf

    """
    sos = cupy.asarray(sos)
    result_type = sos.dtype
    if result_type.kind in 'bui':
        result_type = cupy.float64

    b = cupy.array([1], dtype=result_type)
    a = cupy.array([1], dtype=result_type)
    n_sections = sos.shape[0]
    for section in range(n_sections):
        b = cupy.polymul(b, sos[section, :3])
        a = cupy.polymul(a, sos[section, 3:])
    return b, a


def sos2zpk(sos):
    """
    Return zeros, poles, and gain of a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    The number of zeros and poles returned will be ``n_sections * 2``
    even if some of these are (effectively) zero.

    See Also
    --------
    scipy.signal.sos2zpk

    """
    n_sections = sos.shape[0]
    z = cupy.zeros(n_sections*2, cupy.complex128)
    p = cupy.zeros(n_sections*2, cupy.complex128)
    k = 1.
    for section in range(n_sections):
        # XXX: may just solve a quadratic equation instead of tf2zpk
        zpk = tf2zpk(sos[section, :3], sos[section, 3:])
        z[2*section:2*section + len(zpk[0])] = zpk[0]
        p[2*section:2*section + len(zpk[1])] = zpk[1]
        k *= zpk[2]
    return z, p, k


def tf2ss(num, den):
    r"""Transfer function to state-space representation.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the coefficients of the numerator and
        denominator polynomials, in order of descending degree. The
        denominator needs to be at least as long as the numerator.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.

    See Also
    --------
    scipy.signal.tf2ss
    """
    # Controller canonical state-space representation.
    #  if M+1 = len(num) and K+1 = len(den) then we must have M <= K
    #  states are found by asserting that X(s) = U(s) / D(s)
    #  then Y(s) = N(s) * X(s)
    #
    #   A, B, C, and D follow quite naturally.
    #
    num, den = normalize(num, den)   # Strips zeros, checks arrays
    nn = len(num.shape)
    if nn == 1:
        num = cupy.asarray([num], num.dtype)
    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # Null system
        return (cupy.array([], float),
                cupy.array([], float),
                cupy.array([], float),
                cupy.array([], float))

    # pad numerator to have same number of columns has denominator
    num = cupy.hstack((cupy.zeros((num.shape[0], K - M), num.dtype), num))

    if num.shape[-1] > 0:
        D = cupy.atleast_2d(num[:, 0])

    else:
        # We don't assign it an empty array because this system
        # is not 'null'. It just doesn't have a non-zero D
        # matrix. Thus, it should have a non-zero shape so that
        # it can be operated on by functions like 'ss2tf'
        D = cupy.array([[0]], float)

    if K == 1:
        D = D.reshape(num.shape)

        return (cupy.zeros((1, 1)), cupy.zeros((1, D.shape[1])),
                cupy.zeros((D.shape[0], 1)), D)

    frow = -cupy.array([den[1:]])
    A = cupy.r_[frow, cupy.eye(K - 2, K - 1)]
    B = cupy.eye(K - 1, 1)
    C = num[:, 1:] - cupy.outer(num[:, 0], den[1:])
    D = D.reshape((C.shape[0], B.shape[1]))

    return A, B, C, D


def ss2tf(A, B, C, D, input=0):
    r"""State-space to transfer function.

    A, B, C, D defines a linear state-space system with `p` inputs,
    `q` outputs, and `n` state variables.

    Parameters
    ----------
    A : array_like
        State (or system) matrix of shape ``(n, n)``
    B : array_like
        Input matrix of shape ``(n, p)``
    C : array_like
        Output matrix of shape ``(q, n)``
    D : array_like
        Feedthrough (or feedforward) matrix of shape ``(q, p)``
    input : int, optional
        For multiple-input systems, the index of the input to use.

    Returns
    -------
    num : 2-D ndarray
        Numerator(s) of the resulting transfer function(s). `num` has one row
        for each of the system's outputs. Each row is a sequence representation
        of the numerator polynomial.
    den : 1-D ndarray
        Denominator of the resulting transfer function(s). `den` is a sequence
        representation of the denominator polynomial.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.ss2tf

    """
    # transfer function is C (sI - A)**(-1) B + D

    # Check consistency and make them all rank-2 arrays
    A, B, C, D = abcd_normalize(A, B, C, D)

    nout, nin = D.shape
    if input >= nin:
        raise ValueError("System does not have the input specified.")

    # make SIMO from possibly MIMO system.
    B = B[:, input:input + 1]
    D = D[:, input:input + 1]

    try:
        den = poly(A)
    except ValueError:
        den = 1

    if (prod(B.shape) == 0) and (prod(C.shape) == 0):
        num = cupy.ravel(D)
        if (prod(D.shape) == 0) and (prod(A.shape) == 0):
            den = []
        return num, den

    num_states = A.shape[0]
    type_test = A[:, 0] + B[:, 0] + C[0, :] + D + 0.0
    num = cupy.empty((nout, num_states + 1), type_test.dtype)
    for k in range(nout):
        Ck = cupy.atleast_2d(C[k, :])
        num[k] = poly(A - B @ Ck) + (D[k] - 1) * den

    return num, den


def zpk2ss(z, p, k):
    """Zero-pole-gain representation to state-space representation

    Parameters
    ----------
    z, p : sequence
        Zeros and poles.
    k : float
        System gain.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.

    See Also
    --------
    scipy.signal.zpk2ss

    """
    return tf2ss(*zpk2tf(z, p, k))


def ss2zpk(A, B, C, D, input=0):
    """State-space representation to zero-pole-gain representation.

    A, B, C, D defines a linear state-space system with `p` inputs,
    `q` outputs, and `n` state variables.

    Parameters
    ----------
    A : array_like
        State (or system) matrix of shape ``(n, n)``
    B : array_like
        Input matrix of shape ``(n, p)``
    C : array_like
        Output matrix of shape ``(q, n)``
    D : array_like
        Feedthrough (or feedforward) matrix of shape ``(q, p)``
    input : int, optional
        For multiple-input systems, the index of the input to use.

    Returns
    -------
    z, p : sequence
        Zeros and poles.
    k : float
        System gain.

    See Also
    --------
    scipy.signal.ss2zpk

    """
    return tf2zpk(*ss2tf(A, B, C, D, input=input))


# ### Low-level analog filter prototypes ###

# TODO (ev-br): move to a better place (_filter_design.py (?))

def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.

    See Also
    --------
    butter : Filter design function using this prototype
    scipy.signal.buttap

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = cupy.array([])
    m = cupy.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -cupy.exp(1j * pi * m / (2 * N))
    k = 1
    return z, p, k


def cheb1ap(N, rp):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rp` decibels of ripple in the passband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    cheby1 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero error
        # Even order filters have DC gain of -rp dB
        return cupy.array([]), cupy.array([]), 10**(-rp/20)
    z = cupy.array([])

    # Ripple factor (epsilon)
    eps = cupy.sqrt(10 ** (0.1 * rp) - 1.0)
    mu = 1.0 / N * cupy.arcsinh(1 / eps)

    # Arrange poles in an ellipse on the left half of the S-plane
    m = cupy.arange(-N+1, N, 2)
    theta = pi * m / (2*N)
    p = -cupy.sinh(mu + 1j*theta)

    k = cupy.prod(-p, axis=0).real
    if N % 2 == 0:
        k = k / cupy.sqrt(1 + eps * eps)

    return z, p, k


def cheb2ap(N, rs):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rs` decibels of ripple in the stopband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first reaches ``-rs``.

    See Also
    --------
    cheby2 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        return cupy.array([]), cupy.array([]), 1

    # Ripple factor (epsilon)
    de = 1.0 / cupy.sqrt(10 ** (0.1 * rs) - 1)
    mu = cupy.arcsinh(1.0 / de) / N

    if N % 2:
        m = cupy.concatenate((cupy.arange(-N+1, 0, 2),
                              cupy.arange(2, N, 2)))
    else:
        m = cupy.arange(-N+1, N, 2)

    z = -cupy.conjugate(1j / cupy.sin(m * pi / (2.0 * N)))

    # Poles around the unit circle like Butterworth
    p = -cupy.exp(1j * pi * cupy.arange(-N+1, N, 2) / (2 * N))
    # Warp into Chebyshev II
    p = cupy.sinh(mu) * p.real + 1j * cupy.cosh(mu) * p.imag
    p = 1.0 / p

    k = (cupy.prod(-p, axis=0) / cupy.prod(-z, axis=0)).real
    return z, p, k


# ### Elliptic filter prototype ###

_POW10_LOG10 = math.log(10)


def _pow10m1(x):
    """10 ** x - 1 for x near 0"""
    return cupy.expm1(_POW10_LOG10 * x)


def _ellipdeg(n, m1):
    """Solve degree equation using nomes

    Given n, m1, solve
       n * K(m) / K'(m) = K1(m1) / K1'(m1)
    for m

    See [1], Eq. (49)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """
    # number of terms in solving degree equation
    _ELLIPDEG_MMAX = 7

    K1 = special.ellipk(m1)
    K1p = special.ellipkm1(m1)

    q1 = cupy.exp(-pi * K1p / K1)
    q = q1 ** (1/n)

    mnum = cupy.arange(_ELLIPDEG_MMAX + 1)
    mden = cupy.arange(1, _ELLIPDEG_MMAX + 2)

    num = (q ** (mnum * (mnum+1))).sum()
    den = 1 + 2 * (q ** (mden**2)).sum()

    return 16 * q * (num / den) ** 4


def _arc_jac_sn(w, m):
    """Inverse Jacobian elliptic sn

    Solve for z in w = sn(z, m)

    Parameters
    ----------
    w : complex scalar
        argument

    m : scalar
        modulus; in interval [0, 1]


    See [1], Eq. (56)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

    """
    # Maximum number of iterations in Landen transformation recursion
    # sequence.  10 is conservative; unit tests pass with 4, Orfanidis
    # (see _arc_jac_cn [1]) suggests 5.
    _ARC_JAC_SN_MAXITER = 10

    def _complement(kx):
        # (1-k**2) ** 0.5; the expression below
        # works for small kx
        return ((1 - kx) * (1 + kx)) ** 0.5

    k = m ** 0.5

    if k > 1:
        return cupy.nan
    elif k == 1:
        return cupy.arctanh(w)

    ks = [k]
    niter = 0
    while ks[-1] != 0:
        k_ = ks[-1]
        k_p = _complement(k_)
        ks.append((1 - k_p) / (1 + k_p))
        niter += 1
        if niter > _ARC_JAC_SN_MAXITER:
            raise ValueError('Landen transformation not converging')

    K = cupy.prod(1 + cupy.array(ks[1:])) * pi/2

    wns = [w]

    for kn, knext in zip(ks[:-1], ks[1:]):
        wn = wns[-1]
        wnext = (2 * wn /
                 ((1 + knext) * (1 + _complement(kn * wn))))
        wns.append(wnext)

    u = 2 / pi * cupy.arcsin(wns[-1])

    z = K * u
    return z


def _arc_jac_sc1(w, m):
    """Real inverse Jacobian sc, with complementary modulus

    Solve for z in w = sc(z, 1-m)

    w - real scalar

    m - modulus

    Using that sc(z, m) = -i * sn(i * z, 1 - m)
    cf scipy/signal/_filter_design.py analog for an explanation
    and a reference.

    """

    zcomplex = _arc_jac_sn(1j * w, m)
    if abs(zcomplex.real) > 1e-14:
        raise ValueError

    return zcomplex.imag


def ellipap(N, rp, rs):
    """Return (z,p,k) of Nth-order elliptic analog lowpass filter.

    The filter is a normalized prototype that has `rp` decibels of ripple
    in the passband and a stopband `rs` decibels down.

    The filter's angular (e.g., rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    ellip : Filter design function using this prototype
    scipy.signal.elliap

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        # Even order filters have DC gain of -rp dB
        return cupy.array([]), cupy.array([]), 10**(-rp/20)
    elif N == 1:
        p = -cupy.sqrt(1.0 / _pow10m1(0.1 * rp))
        k = -p
        z = []
        return cupy.asarray(z), cupy.asarray(p), k

    eps_sq = _pow10m1(0.1 * rp)

    eps = cupy.sqrt(eps_sq)
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0:
        raise ValueError("Cannot design a filter with given rp and rs"
                         " specifications.")

    m = _ellipdeg(N, ck1_sq)
    capk = special.ellipk(m)
    j = cupy.arange(1 - N % 2, N, 2)
    EPSILON = 2e-16

    s, c, d, phi = special.ellipj(j * capk / N, m * cupy.ones_like(j))
    snew = cupy.compress(cupy.abs(s) > EPSILON, s, axis=-1)
    z = 1.j / (cupy.sqrt(m) * snew)
    z = cupy.concatenate((z, z.conj()))

    r = _arc_jac_sc1(1. / eps, ck1_sq)
    v0 = capk * r / (N * special.ellipk(ck1_sq))

    sv, cv, dv, phi = special.ellipj(v0, 1 - m)
    p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)

    if N % 2:
        mask = cupy.abs(p.imag) > EPSILON * \
            cupy.sqrt((p * p.conj()).sum(axis=0).real)
        newp = cupy.compress(mask, p, axis=-1)
        p = cupy.concatenate((p, newp.conj()))
    else:
        p = cupy.concatenate((p, p.conj()))

    k = (cupy.prod(-p, axis=0) / cupy.prod(-z, axis=0)).real
    if N % 2 == 0:
        k = k / cupy.sqrt(1 + eps_sq)

    return z, p, k


# ### *ord functions to accopany *ap functions

def _validate_gpass_gstop(gpass, gstop):

    if gpass <= 0.0:
        raise ValueError("gpass should be larger than 0.0")
    elif gstop <= 0.0:
        raise ValueError("gstop should be larger than 0.0")
    elif gpass > gstop:
        raise ValueError("gpass should be smaller than gstop")


def _pre_warp(wp, ws, analog):
    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = cupy.tan(pi * wp / 2.0)
        stopb = cupy.tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0
    return passb, stopb


def _validate_wp_ws(wp, ws, fs, analog):
    wp = cupy.atleast_1d(wp)
    ws = cupy.atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2 * wp / fs
        ws = 2 * ws / fs

    filter_type = 2 * (len(wp) - 1) + 1
    if wp[0] >= ws[0]:
        filter_type += 1

    return wp, ws, filter_type


def _find_nat_freq(stopb, passb, gpass, gstop, filter_type, filter_kind):
    if filter_type == 1:            # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:          # stop
        wp0 = _optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                  args=(0, passb, stopb, gpass, gstop,
                                        filter_kind),
                                  disp=0)
        passb[0] = wp0
        wp1 = _optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                  args=(1, passb, stopb, gpass, gstop,
                                        filter_kind),
                                  disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:          # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))
    else:
        raise ValueError(f"should not happen: {filter_type =}.")

    nat = min(cupy.abs(nat))
    return nat, passb


def _postprocess_wn(WN, analog, fs):
    wn = WN if analog else cupy.arctan(WN) * 2.0 / pi
    if len(wn) == 1:
        wn = wn[0]
    if fs is not None:
        wn = wn * fs / 2
    return wn


def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):
    """
    Band Stop Objective Function for order minimization.

    Returns the non-integer order for an analog band stop filter.

    Parameters
    ----------
    wp : scalar
        Edge of passband `passb`.
    ind : int, {0, 1}
        Index specifying which `passb` edge to vary (0 or 1).
    passb : ndarray
        Two element sequence of fixed passband edges.
    stopb : ndarray
        Two element sequence of fixed stopband edges.
    gstop : float
        Amount of attenuation in stopband in dB.
    gpass : float
        Amount of ripple in the passband in dB.
    type : {'butter', 'cheby', 'ellip'}
        Type of filter.

    Returns
    -------
    n : scalar
        Filter order (possibly non-integer).

    See Also
    --------
    scipy.signal.band_stop_obj

    """

    _validate_gpass_gstop(gpass, gstop)

    passbC = passb.copy()
    passbC[ind] = wp
    nat = (stopb * (passbC[0] - passbC[1]) /
           (stopb ** 2 - passbC[0] * passbC[1]))
    nat = min(cupy.abs(nat))

    if type == 'butter':
        GSTOP = 10 ** (0.1 * cupy.abs(gstop))
        GPASS = 10 ** (0.1 * cupy.abs(gpass))
        n = (cupy.log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * cupy.log10(nat)))
    elif type == 'cheby':
        GSTOP = 10 ** (0.1 * cupy.abs(gstop))
        GPASS = 10 ** (0.1 * cupy.abs(gpass))
        n = cupy.arccosh(
            cupy.sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / cupy.arccosh(nat)
    elif type == 'ellip':
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = cupy.sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        d0 = special.ellipk(cupy.array([arg0 ** 2, 1 - arg0 ** 2]))
        d1 = special.ellipk(cupy.array([arg1 ** 2, 1 - arg1 ** 2]))
        n = (d0[0] * d1[1] / (d0[1] * d1[0]))
    else:
        raise ValueError("Incorrect type: %s" % type)
    return n


def buttord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Butterworth filter order selection.

    Return the order of the lowest order digital or analog Butterworth filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Butterworth filter which meets specs.
    wn : ndarray or float
        The Butterworth natural frequency (i.e. the "3dB frequency"). Should
        be used with `butter` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `butter`.

    See Also
    --------
    scipy.signal.buttord
    butter : Filter design using order and critical points
    cheb1ord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    """
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(
        stopb, passb, gpass, gstop, filter_type, 'butter')

    GSTOP = 10 ** (0.1 * cupy.abs(gstop))
    GPASS = 10 ** (0.1 * cupy.abs(gpass))
    ord = int(cupy.ceil(cupy.log10((GSTOP - 1.0) /
              (GPASS - 1.0)) / (2 * cupy.log10(nat))))

    # Find the Butterworth natural frequency WN (or the "3dB" frequency")
    # to give exactly gpass at passb.
    try:
        W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord))
    except ZeroDivisionError:
        W0 = 1.0
        warnings.warn("Order is zero...check input parameters.",
                      RuntimeWarning, 2)

    # now convert this frequency back from lowpass prototype
    # to the original analog filter

    if filter_type == 1:  # low
        WN = W0 * passb
    elif filter_type == 2:  # high
        WN = passb / W0
    elif filter_type == 3:  # stop
        WN = cupy.empty(2, float)
        discr = cupy.sqrt((passb[1] - passb[0]) ** 2 +
                          4 * W0 ** 2 * passb[0] * passb[1])
        WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
        WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
        WN = cupy.sort(cupy.abs(WN))
    elif filter_type == 4:  # pass
        W0 = cupy.array([-W0, W0], dtype=float)
        WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
              cupy.sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                        passb[0] * passb[1]))
        WN = cupy.sort(cupy.abs(WN))
    else:
        raise ValueError("Bad type: %s" % filter_type)

    wn = _postprocess_wn(WN, analog, fs)

    return ord, wn


def cheb1ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type I filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type I
    filter that loses no more than `gpass` dB in the passband and has at
    least `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type I filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby1` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby1`.

    See Also
    --------
    scipy.signal.cheb1ord
    cheby1 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    """
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(
        stopb, passb, gpass, gstop, filter_type, 'cheby')

    GSTOP = 10 ** (0.1 * cupy.abs(gstop))
    GPASS = 10 ** (0.1 * cupy.abs(gpass))
    v_pass_stop = cupy.arccosh(cupy.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(cupy.ceil(v_pass_stop / cupy.arccosh(nat)))

    # Natural frequencies are just the passband edges
    wn = _postprocess_wn(passb, analog, fs)

    return ord, wn


def cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type II filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type II
    filter that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type II filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby2` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby2`.

    See Also
    --------
    scipy.signal.cheb2ord
    cheby2 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    """
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(
        stopb, passb, gpass, gstop, filter_type, 'cheby')

    GSTOP = 10 ** (0.1 * cupy.abs(gstop))
    GPASS = 10 ** (0.1 * cupy.abs(gpass))
    v_pass_stop = cupy.arccosh(cupy.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(cupy.ceil(v_pass_stop / cupy.arccosh(nat)))

    # Find frequency where analog response is -gpass dB.
    # Then convert back from low-pass prototype to the original filter.

    new_freq = cupy.cosh(1.0 / ord * v_pass_stop)
    new_freq = 1.0 / new_freq

    if filter_type == 1:
        nat = passb / new_freq
    elif filter_type == 2:
        nat = passb * new_freq
    elif filter_type == 3:
        nat = cupy.empty(2, dtype=float)
        nat[0] = (new_freq / 2.0 * (passb[0] - passb[1]) +
                  cupy.sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 +
                            passb[1] * passb[0]))
        nat[1] = passb[1] * passb[0] / nat[0]
    elif filter_type == 4:
        nat = cupy.empty(2, dtype=float)
        nat[0] = (1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) +
                  cupy.sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2)
                            + passb[1] * passb[0]))
        nat[1] = passb[0] * passb[1] / nat[0]

    wn = _postprocess_wn(nat, analog, fs)

    return ord, wn


def ellipord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Elliptic (Cauer) filter order selection.

    Return the order of the lowest order digital or analog elliptic filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    ord : int
        The lowest order for an Elliptic (Cauer) filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `ellip` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `ellip`.

    See Also
    --------
    scipy.signal.ellipord
    ellip : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec
    """
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(
        stopb, passb, gpass, gstop, filter_type, 'ellip')

    arg1_sq = _pow10m1(0.1 * gpass) / _pow10m1(0.1 * gstop)
    arg0 = 1.0 / nat
    d0 = special.ellipk(arg0 ** 2), special.ellipkm1(arg0 ** 2)
    d1 = special.ellipk(arg1_sq), special.ellipkm1(arg1_sq)
    ord = int(cupy.ceil(d0[0] * d1[1] / (d0[1] * d1[0])))

    wn = _postprocess_wn(passb, analog, fs)

    return ord, wn

""" IIR filter conversion utilities.

Split off _filter_design.py
"""
import warnings

import cupy
from cupyx.scipy.special import binom as comb


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
    if 'f' in trim:
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

    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")

    fs2 = 2.0 * fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = cupy.append(z_z, -cupy.ones(degree))

    # Compensate for gain change
    k_z = k * (cupy.prod(fs2 - z) / cupy.prod(fs2 - p)).real

    return z_z, p_z, k_z


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
                for l in range(M - i + 1):
                    if k + l == j:
                        val += comb(i, k) * comb(M - i, l) * bNi * (-1)**k
        bprime[j] = cupy.real(val)

    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            aDi = a[D - i] * (2 * fs)**i
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += comb(i, k) * comb(M - i, l) * aDi * (-1)**k
        aprime[j] = cupy.real(val)

    return normalize(bprime, aprime)

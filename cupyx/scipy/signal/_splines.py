
import cupy
from cupyx.scipy.signal._signaltools import lfilter


def symiirorder1(input, c0, z1, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of first-order sections.  The second section uses a
    reversed sequence.  This implements a system with the following
    transfer function and mirror-symmetric boundary conditions::

                           c0
           H(z) = ---------------------
                   (1-z1/z) (1 - z1 z)

    The resulting signal will have mirror symmetric boundary conditions
    as well.

    Parameters
    ----------
    input : ndarray
        The input signal.
    c0, z1 : scalar
        Parameters in the transfer function.
    precision :
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """

    if cupy.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')

    if precision <= 0.0 or precision > 1.0:
        precision = cupy.finfo(input.dtype).resolution

    precision *= precision
    pos = cupy.arange(0, input.size + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos

    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1[1:] * input) + input[0]

    overflow = False
    if precision == 1.0:
        zi = cum_poly[0]
    else:
        all_valid = diff <= precision
        valid_before = all_valid[1:]
        valid_after = all_valid[:-1]
        valid = cupy.logical_xor(valid_before, valid_after)
        valid_starting = cupy.where(valid, cum_poly, cupy.nan)
        zi = cupy.nanmax(valid_starting, keepdims=True)
        zi_pos = pos[1:][valid]
        overflow = cupy.where(zi_pos >= input.size, True, False)

    if cupy.isnan(zi) or overflow:
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    # Apply first the system 1 / (1 - z1 * z^-1)
    y1, _ = lfilter(
        cupy.ones(1, dtype=input.dtype), cupy.r_[1, -z1], input[1:], zi=zi)
    y1 = cupy.r_[zi, y1]

    # Compute backward symmetric condition and apply the system
    # c0 / (1 - z1 * z)
    zi = -c0 / (z1 - 1.0) * y1[-1]
    out, _ = lfilter(c0, cupy.r_[1, -z1], y1[:-1][::-1], zi=zi)
    return cupy.r_[out[::-1], zi]

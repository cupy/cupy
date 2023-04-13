
import cupy
from cupyx.scipy.signal._signaltools import lfilter


def _find_initial_cond(all_valid, cum_poly, n):
    indices = cupy.where(all_valid)[0] + 1
    zi = cupy.nan
    if indices.size > 0:
        zi = cupy.where(
            indices[0] >= n, cupy.nan, cum_poly[indices[0] - 1])
    return zi


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
    pos = cupy.arange(1, input.size + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos

    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1 * input) + input[0]
    all_valid = diff <= precision

    zi = _find_initial_cond(all_valid, cum_poly, input.size)

    if cupy.isnan(zi):
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


def _compute_symiirorder2_fwd_hc(k, cs, r, omega):
    base = None
    if omega == 0.0:
        base = cs * cupy.power(r, k) * (k+1)
    elif omega == cupy.pi:
        base = cs * cupy.power(r, k) * (k + 1) * (1 - 2 * (k % 2))
    else:
        base = (cs * cupy.power(r, k) * cupy.sin(omega * (k + 1)) /
                cupy.sin(omega))
    return cupy.where(k < 0, 0.0, base)


def _compute_symiirorder2_bwd_hs(k, cs, rsq, omega):
    cssq = cs * cs
    k = cupy.abs(k)
    rsupk = cupy.power(rsq, k / 2.0)

    if omega == 0.0:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq)
        return c0 * rsupk * (1 + gamma * k)

    if omega == cupy.pi:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2))
        return c0 * rsupk * (1 + gamma * k)

    c0 = (cssq * (1.0 + rsq) / (1.0 - rsq) /
          (1 - 2 * rsq * cupy.cos(2 * omega) + rsq * rsq))
    gamma = (1.0 - rsq) / (1.0 + rsq) / cupy.tan(omega)
    return c0 * rsupk * (cupy.cos(omega * k) + gamma * cupy.sin(omega * k))


def symiirorder2(input, r, omega, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of second-order sections.  The second section uses a
    reversed sequence.  This implements the following transfer function::

                                  cs^2
         H(z) = ---------------------------------------
                (1 - a2/z - a3/z^2) (1 - a2 z - a3 z^2 )

    where::

          a2 = 2 * r * cos(omega)
          a3 = - r ** 2
          cs = 1 - 2 * r * cos(omega) + r ** 2

    Parameters
    ----------
    input : ndarray
        The input signal.
    r, omega : float
        Parameters in the transfer function.
    precision : float
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')

    if precision <= 0.0 or precision > 1.0:
        precision = cupy.finfo(input.dtype).resolution

    rsq = r * r
    a2 = 2 * r * cupy.cos(omega)
    a3 = -rsq
    cs = cupy.atleast_1d(1 - 2 * r * cupy.cos(omega) + rsq)

    # First compute the symmetric forward starting conditions
    precision *= precision
    pos = cupy.arange(0, input.size + 2, dtype=input.dtype)
    diff = _compute_symiirorder2_fwd_hc(pos, cs, r, omega)
    err = diff * diff
    cum_poly_y0 = cupy.cumsum(diff[1:-1] * input) + diff[0] * input[0]
    all_valid = err <= precision

    y0 = _find_initial_cond(all_valid[1:-1], cum_poly_y0, input.size)

    if cupy.isnan(y0):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    cum_poly_y1 = (cupy.cumsum(diff[2:] * input) +
                   diff[0] * input[1] + diff[1] * input[0])

    y1 = _find_initial_cond(all_valid[2:], cum_poly_y1, input.size)

    if cupy.isnan(y1):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    # Apply the system cs / (1 - a2 * z^-1 - a3 * z^-2)
    zi = cupy.r_[y0, y1]
    y_fwd, _ = lfilter(cs, cupy.r_[1, -a2, -a3], input[2:], zi=zi)
    y_fwd = cupy.r_[zi, y_fwd]

    # Then compute the symmetric backward starting conditions
    diff = _compute_symiirorder2_bwd_hs(pos, cs, rsq, omega)
    diff_mid = cupy.expand_dims(diff[1:-1], -1)
    diff_exp = cupy.broadcast_to(diff_mid, (diff_mid.shape[0], 2)).ravel()
    diff_exp = cupy.r_[diff[0], diff_exp, diff[-1]]
    diff_exp = cupy.reshape(diff_exp, (diff.size - 1, 2))

    diff = cupy.sum(diff_exp, -1)
    err = diff * diff

    cum_poly_y0 = cupy.cumsum(diff[:-1] * input[::-1])
    all_valid = err <= precision

    y0 = _find_initial_cond(all_valid, cum_poly_y0, input.size)

    if cupy.isnan(y0):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    diff_pos = cupy.c_[pos - 1, pos + 2][:-1]
    diff = _compute_symiirorder2_bwd_hs(diff_pos, cs, rsq, omega)
    diff = diff.sum(axis=-1)
    err = diff * diff

    cum_poly_y1 = cupy.cumsum(diff[:-1] * input[::-1])
    all_valid = err <= precision

    y1 = _find_initial_cond(all_valid, cum_poly_y1, input.size)

    if cupy.isnan(y1):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    # Apply the system cs / (1 - a2 * z^1 - a3 * z^2)
    zi = cupy.r_[y0, y1]
    out, _ = lfilter(cs, cupy.r_[1, -a2, -a3], y_fwd[:-2][::-1], zi=zi)
    return cupy.r_[out[::-1], zi[::-1]]

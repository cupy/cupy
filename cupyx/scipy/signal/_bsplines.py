import cupy
import cupyx.scipy.ndimage


def sepfir2d(input, hrow, hcol):
    """Convolve with a 2-D separable FIR filter.

    Convolve the rank-2 input array with the separable filter defined by the
    rank-1 arrays hrow, and hcol. Mirror symmetric boundary conditions are
    assumed. This function can be used to find an image given its B-spline
    representation.

    The arguments `hrow` and `hcol` must be 1-dimensional and of off length.

    Args:
        input (cupy.ndarray): The input signal
        hrow (cupy.ndarray): Row direction filter
        hcol (cupy.ndarray): Column direction filter

    Returns:
        cupy.ndarray: The filtered signal

    .. seealso:: :func:`scipy.signal.sepfir2d`
    """
    if any(x.ndim != 1 or x.size % 2 == 0 for x in (hrow, hcol)):
        raise ValueError('hrow and hcol must be 1 dimensional and odd length')
    dtype = input.dtype
    if dtype.kind == 'c':
        dtype = cupy.complex64 if dtype == cupy.complex64 else cupy.complex128
    elif dtype == cupy.float32 or dtype.itemsize <= 2:
        dtype = cupy.float32
    else:
        dtype = cupy.float64
    input = input.astype(dtype, copy=False)
    hrow = hrow.astype(dtype, copy=False)
    hcol = hcol.astype(dtype, copy=False)
    filters = (hcol[::-1].conj(), hrow[::-1].conj())
    return cupyx.scipy.ndimage._filters._run_1d_correlates(
        input, (0, 1), lambda i: filters[i], None, 'reflect', 0)


def _coeff_smooth(lam):
    xi = 1 - 96 * lam + 24 * lam * cupy.sqrt(3 + 144 * lam)
    omeg = cupy.arctan2(cupy.sqrt(144 * lam - 1), cupy.sqrt(xi))
    rho = (24 * lam - 1 - cupy.sqrt(xi)) / (24 * lam)
    rho = rho * cupy.sqrt(
        (48 * lam + 24 * lam * cupy.sqrt(3 + 144 * lam)) / xi)
    return rho, omeg


def _hc(k, cs, rho, omega):
    return (cs / cupy.sin(omega) * (rho ** k) * cupy.sin(omega * (k + 1)) *
            cupy.greater(k, -1))


def _hs(k, cs, rho, omega):
    c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
          (1 - 2 * rho * rho * cupy.cos(2 * omega) + rho ** 4))
    gamma = (1 - rho * rho) / (1 + rho * rho) / cupy.tan(omega)
    ak = abs(k)
    return c0 * rho ** ak * (
        cupy.cos(omega * ak) + gamma * cupy.sin(omega * ak))


def _cubic_smooth_coeff(signal, lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cupy.cos(omega) + rho * rho
    K = len(signal)
    yp = cupy.zeros((K,), signal.dtype.char)
    k = cupy.arange(K)
    yp[0] = (_hc(0, cs, rho, omega) * signal[0] +
             cupy.sum(_hc(k + 1, cs, rho, omega) * signal))

    yp[1] = (_hc(0, cs, rho, omega) * signal[0] +
             _hc(1, cs, rho, omega) * signal[1] +
             cupy.sum(_hc(k + 2, cs, rho, omega) * signal))

    for n in range(2, K):
        yp[n] = (cs * signal[n] + 2 * rho * cupy.cos(omega) * yp[n - 1] -
                 rho * rho * yp[n - 2])

    y = cupy.zeros((K,), signal.dtype.char)

    y[K - 1] = cupy.sum((_hs(k, cs, rho, omega) +
                         _hs(k + 1, cs, rho, omega)) * signal[::-1])
    y[K - 2] = cupy.sum((_hs(k - 1, cs, rho, omega) +
                         _hs(k + 2, cs, rho, omega)) * signal[::-1])

    for n in range(K - 3, -1, -1):
        y[n] = (cs * yp[n] + 2 * rho * cupy.cos(omega) * y[n + 1] -
                rho * rho * y[n + 2])

    return y


def _cubic_coeff(signal):
    zi = -2 + cupy.sqrt(3)
    K = len(signal)
    yplus = cupy.zeros((K,), signal.dtype.char)
    powers = zi ** cupy.arange(K)
    yplus[0] = signal[0] + zi * sum(powers * signal)
    for k in range(1, K):
        yplus[k] = signal[k] + zi * yplus[k - 1]
    output = cupy.zeros((K,), signal.dtype)
    output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for k in range(K - 2, -1, -1):
        output[k] = zi * (output[k + 1] - yplus[k])
    return output * 6.0


def cspline1d(signal, lamb=0.0):
    """
    Compute cubic spline coefficients for rank-1 array.

    Find the cubic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.

    Returns
    -------
    c : ndarray
        Cubic spline coefficients.

    See Also
    --------
    cspline1d_eval : Evaluate a cubic spline at the new set of points.

    """
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        return _cubic_coeff(signal)

import cupyx.scipy.ndimage


def sepfir2d(input, hrow, hcol):
    """Convolve with a 2-D separable FIR filter.

    Convolve the rank-2 input array with the separable filter defined by the
    rank-1 arrays hrow, and hcol. Mirror symmetric boundary conditions are
    assumed. This function can be used to find an image given its B-spline
    representation.

    Args:
        input (cupy.ndarray): The input signal
        hrow (cupy.ndarray): Row direction filter
        hcol (cupy.ndarray): Column direction filter

    Returns:
        cupy.ndarray: The filtered signal

    .. seealso:: :func:`scipy.signal.sepfir2d`
    """
    filters = (hcol[::-1], hrow[::-1])
    return cupyx.scipy.ndimage.filters._run_1d_correlates(
        input, (0, 1), lambda i: filters[i], None, 'reflect', 0.0)

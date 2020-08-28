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
        # TODO: adding support for complex types requires ndimage filters
        # to support complex types (which they could easily if not for the
        # scipy compatibility requirement of forbidding complex and using
        # float64 intermediates)
        raise TypeError("complex types not currently supported")
    if dtype == cupy.float32 or dtype.itemsize <= 2:
        dtype = cupy.float32
    else:
        dtype = cupy.float64
    input = input.astype(dtype, copy=False)
    hrow = hrow.astype(dtype, copy=False)
    hcol = hcol.astype(dtype, copy=False)
    filters = (hcol[::-1], hrow[::-1])
    return cupyx.scipy.ndimage.filters._run_1d_correlates(
        input, (0, 1), lambda i: filters[i], None, 'reflect', 0)

import cupy


def polyvander(x, deg):
    """Computes the Vandermonde matrix of given degree.

    Args:
        x (cupy.ndarray): array of points
        deg (int): degree of the resulting matrix.

    Returns:
        cupy.ndarray: The Vandermonde matrix

    .. seealso:: :func:`numpy.polynomial.polynomial.polyvander`

    """
    deg.__index__()
    if deg < 0:
        raise ValueError('deg must be non-negative')
    if not isinstance(x, cupy.ndarray) or x.ndim != 1:
        x = cupy.array(x, copy=False, ndmin=1)
    x = x + 0.0
    dims = (deg + 1,) + x.shape
    v = cupy.ones(dims, dtype=x.dtype)
    for i in range(1, deg + 1):
        cupy.multiply(v[i-1], x, v[i])
    return cupy.moveaxis(v, 0, -1)

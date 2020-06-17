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
    deg = cupy.polynomial.polyutils._deprecate_as_int(deg, 'deg')
    if deg < 0:
        raise ValueError('degree must be non-negative')
    if x.ndim == 0:
        x = cupy.expand_dims(x, axis=0)
    x = cupy.expand_dims(x, axis=-1)
    x = x + 0.0
    v = x ** cupy.arange(deg + 1)
    return cupy.asarray(v, x.dtype, order='F')


def polycompanion(c):
    """Computes the companion matrix of c.

    Args:
        c (cupy.ndarray): 1-D array of polynomial coefficients
            ordered from low to high degree.

    Returns:
        cupy.ndarray: Companion matrix of dimensions (deg, deg).

    .. seealso:: :func:`numpy.polynomial.polynomial.polycompanion`

    """
    [c] = cupy.polynomial.polyutils.as_series([c])
    deg = c.size - 1
    if deg == 0:
        raise ValueError('Series must have maximum degree of at least 1.')
    matrix = cupy.eye(deg, k=-1, dtype=c.dtype)
    matrix[:, -1] -= c[:-1] / c[-1]
    return matrix

import cupy


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

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
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    deg = len(c) - 1
    matrix = cupy.zeros((deg, deg), dtype=c.dtype)
    matrix.reshape(-1)[deg:: len(c)] = 1
    matrix[:, -1].__isub__(cupy.divide(c[:-1], c[-1]))
    return matrix

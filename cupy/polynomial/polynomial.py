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
    if not x.ndim:
        x = cupy.expand_dims(x, axis=0)
    x = cupy.expand_dims(x, axis=-1)
    x = x + 0.0
    v = x ** cupy.arange(deg + 1)
    return cupy.asarray(v, x.dtype, order='F')

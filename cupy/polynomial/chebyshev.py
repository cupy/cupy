__all__ = ['chebadd', 'chebsub', 'chebmul', 'chebmulx', 'chebpow']

import functools

import cupy


def _wraps_chebroutine(func):
    def _get_coeffs(x):
        if isinstance(x, cupy.poly1d):
            x = x._coeffs
        elif cupy.isscalar(x):
            x = cupy.atleast_1d(x)
        if isinstance(x, cupy.ndarray):
            if x.size == 0:
                raise ValueError('Coefficient array is empty')
            if x.ndim > 1:
                raise ValueError('Coefficient array is not 1-d')
            if x.dtype.kind == 'b':
                raise ValueError('bool inputs are not allowed')
            return x.ravel()
        raise TypeError('Unsupported type')

    def wrapper(*args):
        coeffs = [_get_coeffs(x) for x in args]
        dtype = cupy.common_type(*coeffs)
        coeffs = [c.astype(dtype, copy=False) for c in coeffs]
        return cupy.polynomial.polyutils.trimseq(func(*coeffs))

    return functools.update_wrapper(wrapper, func)


def _cseries_to_zseries(c):
    zs = cupy.zeros(2 * c.size - 1, c.dtype)
    zs[c.size - 1:] = c / 2
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    n = (zs.size + 1) // 2
    c = zs[n - 1:]
    c[1: n] *= 2
    return c


@_wraps_chebroutine
def chebadd(c1, c2):
    """Adds two Chebyshev series.

    Args:
        c1 (scalar, cupy.ndarray or cupy.poly1d): first chebyshev series input
        c2 (scalar, cupy.ndarray or cupy.poly1d): second chebyshev series input

    Returns:
        cupy.ndarray: the sum of two chebyshev series.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebadd`

    """
    return cupy.polynomial.polyutils._add(c1, c2)


@_wraps_chebroutine
def chebsub(c1, c2):
    """Subtracts a Chebyshev series from another.

    Args:
        c1 (scalar, cupy.ndarray or cupy.poly1d): first chebyshev series input
        c2 (scalar, cupy.ndarray or cupy.poly1d): second chebyshev series input

    Returns:
        cupy.ndarray: the difference of two chebyshev series.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebsub`

    """
    return cupy.polynomial.polyutils._sub(c1, c2)


@_wraps_chebroutine
def chebmulx(c):
    """Multiplies a chebyshev series with the independent variable x.

    Args:
        c (scalar, cupy.ndarray or cupy.poly1d): chebyshev series input.

    Returns:
        cupy.ndarray: the product of a chebyshev series with x.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebmulx`

    """
    if c.size == 1 and c[0] == 0:
        return c
    prd = cupy.empty_like(c, shape=c.size + 1)
    prd[0] = cupy.zeros_like(c[0])
    prd[1] = c[0]
    prd[2:] = c[1:] / 2
    prd[: -2] += c[1:] / 2
    return prd


@_wraps_chebroutine
def chebmul(c1, c2):
    """Multiplies two Chebyshev series.

    Args:
        c1 (scalar, cupy.ndarray or cupy.poly1d): first chebyshev series input
        c2 (scalar, cupy.ndarray or cupy.poly1d): second chebyshev series input

    Returns:
        cupy.ndarray: the product of two chebyshev series.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebmul`

    """
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    return _zseries_to_cseries(cupy.convolve(z1, z2))


def chebpow(c, pow, maxpower=16):
    """Raises a Chebyshev series to a power.

    Args:
        c (scalar, cupy.ndarray or cupy.poly1d): chebyshev series input.
        pow (int): power to which the series is raised.
        maxpower (int, optional): maximum power allowed.

    Returns:
        cupy.ndarray: chebyshev series raised to the required power.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebpow`
    """
    if isinstance(c, cupy.poly1d):
        c = c._coeffs
    elif cupy.isscalar(c):
        c = cupy.atleast_1d(c)
    [c] = cupy.polynomial.polyutils.as_series([c])
    _pow = int(pow)
    if _pow != pow or _pow < 0:
        raise ValueError('Power must be a non-negative integer.')
    if maxpower is not None and _pow > maxpower:
        raise ValueError('Power is too large')
    if _pow == 0:
        return cupy.ones((1,), c.dtype)
    if _pow == 1:
        return c
    zs = _cseries_to_zseries(c)
    return _zseries_to_cseries(cupy.lib._routines_poly._polypow(zs, _pow))

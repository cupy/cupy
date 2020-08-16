__all__ = ['chebdomain', 'chebzero', 'chebone', 'chebx', 'chebadd',
           'chebsub', 'chebmul', 'chebmulx', 'chebpow']

import functools

import cupy


def _wraps_chebroutine(func):
    def wrapper(*args):
        [c1, *c2] = cupy.polynomial.polyutils.as_series([x for x in args])
        return cupy.polynomial.polyutils.trimseq(func(c1, *c2))
    return functools.update_wrapper(wrapper, func)


def _cseries_to_zseries(c):
    n = c.size
    zs = cupy.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1:] = c / 2
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    n = (zs.size + 1) // 2
    c = zs[n - 1:].copy()
    c[1:n] *= 2
    return c


# Chebyshev default domain.
chebdomain = cupy.array([-1, 1])

# Chebyshev coefficients representing zero.
chebzero = cupy.array([0])

# Chebyshev coefficients representing one.
chebone = cupy.array([1])

# Chebyshev coefficients representing the identity x.
chebx = cupy.array([0, 1])


@_wraps_chebroutine
def chebadd(c1, c2):
    """Adds two Chebyshev series.

    Args:
        c1 (cupy.ndarray): first chebyshev series input.
        c2 (cupy.ndarray): second chebyshev series input.

    Returns:
        cupy.ndarray: the sum of two chebyshev series.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebadd`

    """
    return cupy.polynomial.polyutils._add(c1, c2)


@_wraps_chebroutine
def chebsub(c1, c2):
    """Subtracts a Chebyshev series from another.

    Args:
        c1 (cupy.ndarray): first chebyshev series input.
        c2 (cupy.ndarray): second chebyshev series input.

    Returns:
        cupy.ndarray: the difference of two chebyshev series.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebsub`

    """
    return cupy.polynomial.polyutils._sub(c1, c2)


@_wraps_chebroutine
def chebmulx(c):
    """Multiplies a chebyshev series with the independent variable x.

    Args:
        c (cupy.ndarray): chebyshev series input.

    Returns:
        cupy.ndarray: the product of a chebyshev series with x.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebmulx`

    """
    if c.size == 1 and c[0] == 0:
        return c
    prd = cupy.empty(c.size + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    if c.size > 1:
        tmp = c[1:] / 2
        prd[2:] = tmp
        prd[0: -2] += tmp
    return prd


@_wraps_chebroutine
def chebmul(c1, c2):
    """Multiplies two Chebyshev series.

    Args:
        c1 (cupy.ndarray): first chebyshev series input.
        c2 (cupy.ndarray): second chebyshev series input.

    Returns:
        cupy.ndarray: the product of two chebyshev series.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebmul`

    """
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = cupy.convolve(z1, z2)
    return _zseries_to_cseries(prd)


def chebpow(c, pow, maxpower=16):
    """Raises a Chebyshev series to a power.

    Args:
        c (cupy.ndarray): chebyshev series input.
        pow (int): power to which the series is raised.
        maxpower (int, optional): maximum power allowed.

    Returns:
        cupy.ndarray: chebyshev series raised to the required power.

    .. seealso:: :func:`numpy.polynomial.chebyshev.chebpow`
    """

    [c] = cupy.polynomial.polyutils.as_series([c])
    _pow = int(pow)
    if _pow != pow or _pow < 0:
        raise ValueError('Power must be a non-negative integer.')
    if maxpower is not None and _pow > maxpower:
        raise ValueError('Power is too large')
    if _pow == 0:
        return cupy.array([1], dtype=c.dtype)
    if _pow == 1:
        return c
    zs = _cseries_to_zseries(c)
    prd = cupy.lib._routines_poly._polypow(zs)
    # for i in range(2, _pow + 1):
    #     prd = cupy.convolve(prd, zs)
    return _zseries_to_cseries(prd)

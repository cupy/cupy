import cupy
from cupy._core import internal


def polyvander(x, deg):
    """Computes the Vandermonde matrix of given degree.

    Args:
        x (cupy.ndarray): array of points
        deg (int): degree of the resulting matrix.

    Returns:
        cupy.ndarray: The Vandermonde matrix

    .. seealso:: :func:`numpy.polynomial.polynomial.polyvander`

    """
    deg = cupy.polynomial.polyutils._as_int(deg, 'deg')
    if deg < 0:
        raise ValueError('degree must be non-negative')
    if x.ndim == 0:
        x = x.ravel()
    dtype = cupy.float64 if x.dtype.kind in 'biu' else x.dtype
    out = x ** cupy.arange(deg + 1, dtype=dtype).reshape((-1,) + (1,) * x.ndim)
    return cupy.moveaxis(out, 0, -1)


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


def polyval(x, c, tensor=True):
    """
    Evaluate a polynomial at points x.

    If `c` is of length `n + 1`, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    numpy.polynomial.polynomial.polyval

    Notes
    -----
    The evaluation uses Horner's method.

    """
    c = cupy.array(c, ndmin=1, copy=False)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = cupy.asarray(x)
    if isinstance(x, cupy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    c0 = c[-1] + x*0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0*x
    return c0


def polyvalfromroots(x, r, tensor=True):
    """
    Evaluate a polynomial specified by its roots at points x.

    If `r` is of length `N`, this function returns the value

    .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `r`.

    If `r` is a 1-D array, then `p(x)` will have the same shape as `x`.  If `r`
    is multidimensional, then the shape of the result depends on the value of
    `tensor`. If `tensor` is ``True`` the shape will be r.shape[1:] + x.shape;
    that is, each polynomial is evaluated at every value of `x`. If `tensor` is
    ``False``, the shape will be r.shape[1:]; that is, each polynomial is
    evaluated only for the corresponding broadcast value of `x`. Note that
    scalars have shape (,).

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `r`.
    r : array_like
        Array of roots. If `r` is multidimensional the first index is the
        root index, while the remaining indices enumerate multiple
        polynomials. For instance, in the two dimensional case the roots
        of each polynomial may be thought of as stored in the columns of `r`.
    tensor : boolean, optional
        If True, the shape of the roots array is extended with ones on the
        right, one for each dimension of `x`. Scalars have dimension 0 for this
        action. The result is that every column of coefficients in `r` is
        evaluated for every element of `x`. If False, `x` is broadcast over the
        columns of `r` for the evaluation.  This keyword is useful when `r` is
        multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    numpy.polynomial.polynomial.polyvalfroomroots
    """
    r = cupy.array(r, ndmin=1, copy=False)
    if r.dtype.char in '?bBhHiIlLqQpP':
        r = r.astype(cupy.double)
    if isinstance(x, (tuple, list)):
        x = cupy.asarray(x)
    if isinstance(x, cupy.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,)*x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")
    return cupy.prod(x - r, axis=0)


def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """Integrate a polynomial.

    Returns the polynomial coefficients c integrated m times from
    lbnd along axis. At each iteration the resulting series is
    *multiplied* by scl and an integration constant, k, is added.
    The scaling factor is for use in a linear change of variable.

    Args:
        c (cupy.ndarray): 1-D array of polynomial coefficients, ordered
            from low to high.
        m (int, optional): Order of integration, must be positive.
            Default is 1.
        k ({[], list, scalar}, optional): Integration constant(s). The value
            of the first integral at zero is the first value in the list, the
            value of the second integral at zero is the second value, etc. If
            k == [] (the default), all constants are set to zero. If m == 1, a
            single scalar can be given instead of a list.
        lbnd (scalar, optional): The lower bound of the integral. Default is 0.
        scl (scalar, optional): Following each integration the result is
            multiplied by scl before the integration constant is added.
            Default is 1.
        axis (int, optional): Axis over which the integral is taken.
            Default is 0.

    Returns:
        cupy.ndarray: Coefficient array of the integral.

    .. seealso:: :func:`numpy.polynomial.polynomial.polyint`

    """
    import numpy  # Only for iterable function

    c = cupy.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    cdt = c.dtype
    if not numpy.iterable(k):
        k = [k]
    cnt = cupy.polynomial.polyutils._as_int(m, "the order of integration")
    iaxis = cupy.polynomial.polyutils._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if cupy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if cupy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = internal._normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0] * (cnt - len(k))
    c = cupy.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and cupy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = cupy.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = cupy.moveaxis(c, 0, iaxis)
    return c


def polyder(c, m=1, scl=1, axis=0):
    """Differentiates a polynomial.

    Args:
        c (cupy.ndarray): Array of polynomial coefficients. If c is
            multidimensional the different axis correspond to different
            variables with the degree in each axis given by the corresponding
            index.
        m (int, optional): Number of derivatives taken, must be non-negative.
            Default is 1.
        scl (scalar, optional): Each differentiation is multiplied by `scl`.
            The end result is multiplication by ``scl**m``. Default is 1.
        axis (int, optional): Axis over which the derivative is taken.
            Default is 0.

    Returns:
        cupy.ndarray: Array of polynomial coefficients of the derivative.

    .. seealso:: :func:`numpy.polynomial.polynomial.polyder`

    """
    c = cupy.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    cdt = c.dtype
    cnt = cupy.polynomial.polyutils._as_int(m, "the order of derivation")
    iaxis = cupy.polynomial.polyutils._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = internal._normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = cupy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = cupy.empty((n,) + c.shape[1:], dtype=cdt)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = cupy.moveaxis(c, 0, iaxis)
    return c

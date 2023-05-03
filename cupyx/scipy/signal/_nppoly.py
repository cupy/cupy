"""Bits and pieces from new-style NumPy polynomial functionality."""
import cupy


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
    polyval2d, polygrid2d, polyval3d, polygrid3d
    numpy.polynomial.polynomial.polyval

    Notes
    -----
    The evaluation uses Horner's method.

    Examples
    --------
    >>> from numpy.polynomial.polynomial import polyval
    >>> polyval(1, [1,2,3])
    6.0
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyval(a, [1,2,3])
    array([[ 1.,   6.],
           [17.,  34.]])
    >>> coef = np.arange(4).reshape(2,2) # multidimensional coefficients
    >>> coef
    array([[0, 1],
           [2, 3]])
    >>> polyval([1,2], coef, tensor=True)
    array([[2.,  4.],
           [4.,  7.]])
    >>> polyval([1,2], coef, tensor=False)
    array([2.,  7.])

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

    .. versionadded:: 1.12

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
    polyroots, polyfromroots, polyval

    Examples
    --------
    >>> from numpy.polynomial.polynomial import polyvalfromroots
    >>> polyvalfromroots(1, [1,2,3])
    0.0
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyvalfromroots(a, [-1, 0, 1])
    array([[-0.,   0.],
           [ 6.,  24.]])
    >>> r = np.arange(-2, 2).reshape(2,2) # multidimensional coefficients
    >>> r # each column of r defines one polynomial
    array([[-2, -1],
           [ 0,  1]])
    >>> b = [-2, 1]
    >>> polyvalfromroots(b, r, tensor=True)
    array([[-0.,  3.],
           [ 3., 0.]])
    >>> polyvalfromroots(b, r, tensor=False)
    array([-0.,  0.])
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

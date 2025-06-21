import cupy

import operator


def _as_int(x, desc):
    """
    Like `operator.index`, but emits a custom exception when passed an
    incorrect type

    Parameters
    ----------
    x : int-like
        Value to interpret as an integer
    desc : str
        description to include in any error message

    Raises
    ------
    TypeError : if x is a float or non-numeric
    """
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{desc} must be an integer, received {x}") from e


def trimseq(seq):
    """Removes small polynomial series coefficients.

    Args:
        seq (cupy.ndarray): input array.

    Returns:
        cupy.ndarray: input array with trailing zeros removed. If the
        resulting output is empty, it returns the first element.

    .. seealso:: :func:`numpy.polynomial.polyutils.trimseq`

    """
    if seq.ndim == 0:
        raise TypeError('Input must be 1-d array')
    if seq.ndim > 1:
        raise ValueError('Input must be 1-d array')
    if seq.size == 0:
        return seq
    ret = cupy.trim_zeros(seq, trim='b')
    if ret.size > 0:
        return ret
    return seq[:1]


def as_series(alist, trim=True):
    """Returns argument as a list of 1-d arrays.

    Args:
        alist (cupy.ndarray or list of cupy.ndarray): 1-D or 2-D input array.
        trim (bool, optional): trim trailing zeros.

    Returns:
        list of cupy.ndarray: list of 1-D arrays.

    .. seealso:: :func:`numpy.polynomial.polyutils.as_series`

    """
    arrays = []
    for a in alist:
        if a.size == 0:
            raise ValueError('Coefficient array is empty')
        if a.ndim > 1:
            raise ValueError('Coefficient array is not 1-d')
        if a.dtype.kind == 'b':
            raise ValueError('Coefficient arrays have no common type')
        a = a.ravel()
        if trim:
            a = trimseq(a)
        arrays.append(a)
    dtype = cupy.common_type(*arrays)
    ret = [a.astype(dtype, copy=False) for a in arrays]
    return ret


def trimcoef(c, tol=0):
    """Removes small trailing coefficients from a polynomial.

    Args:
        c(cupy.ndarray): 1d array of coefficients from lowest to highest order.
        tol(number, optional): trailing coefficients whose absolute value are
            less than or equal to ``tol`` are trimmed.

    Returns:
        cupy.ndarray: trimmed 1d array.

    .. seealso:: :func:`numpy.polynomial.polyutils.trimcoef`

    """
    if tol < 0:
        raise ValueError('tol must be non-negative')
    if c.size == 0:
        raise ValueError('Coefficient array is empty')
    if c.ndim > 1:
        raise ValueError('Coefficient array is not 1-d')
    if c.dtype.kind == 'b':
        raise ValueError('bool inputs are not allowed')
    if c.ndim == 0:
        c = c.ravel()
    c = c.astype(cupy.common_type(c), copy=False)
    filt = (cupy.abs(c) > tol)[::-1]
    ind = c.size - cupy._manipulation.add_remove._first_nonzero_krnl(
        filt, c.size).item()
    if ind == 0:
        return cupy.zeros_like(c[:1])
    return c[: ind]


def getdomain(x):
    """Return a domain suitable for given abscissae.

    Args:
        x (array_like): 1-d array of abscissae whose domain will be
            determined.

    Returns:
        cupy.ndarray: 1-d array containing two values. If the inputs are
        complex, then the two returned points are the lower left and upper
        right corners of the smallest rectangle in the complex plane
        containing the points `x`. If the inputs are real, then the
        two points are the ends of the smallest interval containing the
        points `x`.

    .. seealso:: :func:`numpy.polynomial.polyutils.getdomain`

    """
    [x] = as_series([x], trim=False)
    if x.dtype.kind == 'c':  # 'c' for complex
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return cupy.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        return cupy.array((x.min(), x.max()))

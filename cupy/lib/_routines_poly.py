import functools

import numpy

import cupy


def _wraps_polyroutine(func):
    def _get_coeffs(x):
        if isinstance(x, cupy.poly1d):
            return x._coeffs
        if isinstance(x, cupy.ndarray) or cupy.isscalar(x):
            return cupy.atleast_1d(x)
        raise TypeError('Unsupported type')

    def wrapper(*args):
        coeffs = [_get_coeffs(x) for x in args]
        out = func(*coeffs)

        if all([not isinstance(x, cupy.poly1d) for x in args]):
            return out
        if isinstance(out, cupy.ndarray):
            return cupy.poly1d(out)
        if isinstance(out, tuple):
            return tuple([cupy.poly1d(x) for x in out])
        assert False  # Never reach

    return functools.update_wrapper(wrapper, func)


@_wraps_polyroutine
def polyadd(a1, a2):
    """Computes the sum of two polynomials.

    Args:
        a1 (scalar, cupy.ndarray or cupy.poly1d): first input polynomial.
        a2 (scalar, cupy.ndarray or cupy.poly1d): second input polynomial.

    Returns:
        cupy.ndarray or cupy.poly1d: The sum of the inputs.

    .. seealso:: :func:`numpy.polyadd`

    """
    if a1.size < a2.size:
        a1, a2 = a2, a1
    out = cupy.pad(a2, (a1.size - a2.size, 0))
    out = out.astype(cupy.result_type(a1, a2), copy=False)
    out += a1
    return out


@_wraps_polyroutine
def polysub(a1, a2):
    """Computes the difference of two polynomials.

    Args:
        a1 (scalar, cupy.ndarray or cupy.poly1d): first input polynomial.
        a2 (scalar, cupy.ndarray or cupy.poly1d): second input polynomial.

    Returns:
        cupy.ndarray or cupy.poly1d: The difference of the inputs.

    .. seealso:: :func:`numpy.polysub`

    """
    if a1.shape[0] <= a2.shape[0]:
        out = cupy.pad(a1, (a2.shape[0] - a1.shape[0], 0))
        out = out.astype(cupy.result_type(a1, a2), copy=False)
        out -= a2
    else:
        out = cupy.pad(a2, (a1.shape[0] - a2.shape[0], 0))
        out = out.astype(cupy.result_type(a1, a2), copy=False)
        out -= 2 * out - a1
    return out


def polyval(p, x):
    """Evaluates a polynomial at specific values.

    Args:
        p (cupy.ndarray or cupy.poly1d): input polynomial.
        x (scalar, cupy.ndarray): values at which the polynomial
        is evaluated.

    Returns:
        cupy.ndarray or cupy.poly1d: polynomial evaluated at x.

    .. warning::

        This function doesn't currently support poly1d objects nor
        multidimensional ndarrays as values used in evaluation.

    .. seealso:: :func:`numpy.polyval`

    """
    if isinstance(p, cupy.poly1d):
        p = p.coeffs
    if cupy.isscalar(p) or p.ndim == 0:
        raise TypeError('p can be 1d ndarray or poly1d object only')
    if p.ndim != 1:
        # to be consistent with polyarithmetic routines' behavior of
        # not allowing multidimensional polynomial inputs.
        raise ValueError('p can be 1d ndarray or poly1d object only')
    # TODO(Dahlia-Chehata): Support poly1d and multidimensional x
    if isinstance(x, cupy.poly1d) or (isinstance(x, cupy.ndarray)
                                      and x.ndim > 1):
        raise NotImplementedError('poly1d or non 1d values are not'
                                  ' currently supported')
    val = cupy.asarray(x).reshape(-1, 1)
    exp = cupy.tile(cupy.arange(p.size), (val.size, 1))
    out = p[::-1] * cupy.power(val, exp)
    out = out.sum(axis=1)
    dtype = cupy.result_type(p, val)
    # For case: when p is of shape (0,) and x is (), output is
    #  of single valued array to match NumPy's behavior
    if isinstance(x, cupy.ndarray) and x.ndim == 0 and p.size == 0:
        return out[0].astype(dtype, copy=False)
    if cupy.isscalar(x) or x.ndim == 0:
        return dtype.type(out.item())
    # To handle mixed integer and float dtypes combinations for inputs,
    # output should be cast according to NumPy's promotion rules.
    if p.dtype.kind in 'c' or (issubclass(
            val.dtype.type, numpy.integer) and issubclass(
            p.dtype.type, numpy.floating)):
        return out.astype(dtype, copy=False)
    # To handle bool values used in evaluation, casting the output
    # to polynomial dtype is required to match NumPy's results.
    if val.dtype.kind in 'b':
        return out.astype(p.dtype, copy=False)
    return out.astype(val.dtype, copy=False)

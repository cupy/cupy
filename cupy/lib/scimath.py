import cupy


__all__ = [
    'sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin',
    'arctanh'
]


def _tocomplex(arr):
    """Convert its input `arr` to a complex array.
    The input is returned as a complex array of the smallest type that will fit
    the original data: types like single, byte, short, etc. become csingle,
    while others become cdouble.
    A copy of the input is always made.

    Parameters
    ----------
    arr : cupy.ndarray
          Input array.

    Returns
    -------
    array : cupy.ndarray
            An array with the same input data as the input but in complex form.

    Examples
    --------
    First, consider an input of type short:
    >>> a = cupy.array([1,2,3], cupy.short)
    >>> ac = cupy.emath._tocomplex(a); ac
    array([1.+0.j, 2.+0.j, 3.+0.j], dtype=complex64)
    >>> ac.dtype
    dtype('complex64')
    If the input is of type double, the output is correspondingly of the
    complex double type as well:
    >>> b = cupy.array([1,2,3], cupy.double)
    >>> bc = cupy.emath._tocomplex(b); bc
    array([1.+0.j, 2.+0.j, 3.+0.j])
    >>> bc.dtype
    dtype('complex128')
    Note that even if the input was complex to begin with, a copy is still
    made, since the astype() method always copies:
    >>> c = cupy.array([1,2,3], cupy.csingle)
    >>> cc = cupy.lib.scimath._tocomplex(c); cc
    array([1.+0.j, 2.+0.j, 3.+0.j], dtype=complex64)
    >>> cc.dtype
    dtype('complex64')
    >>> c *= 2; c
    array([2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)
    >>> cc
    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)

    See Also
    --------
    numpy.lib.scimath._tocomplex

    """
    if issubclass(arr.dtype.type,
                  (cupy.single, cupy.byte,
                   cupy.short, cupy.ubyte,
                   cupy.ushort, cupy.csingle)):
        return arr.astype(cupy.csingle)
    else:
        return arr.astype(cupy.cdouble)


def _fix_real_lt_zero(x):
    """Convert `x` to complex if it has real, negative components.
    Otherwise, output is just the array version of the input
    (via cupy.asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array : cupy.ndarray

    Examples
    --------
    >>> cupy.lib.scimath._fix_real_lt_zero([1,2])
    array([1, 2])
    >>> cupy.lib.scimath._fix_real_lt_zero([-1,2])
    array([-1.+0.j,  2.+0.j])

    See Also
    --------
    numpy.lib.scimath._fix_real_lt_zero

    """
    x = cupy.asarray(x)
    if cupy.any(cupy.isreal(x) & (x < 0)):
        x = _tocomplex(x)
    return x


def _fix_int_lt_zero(x):
    """Convert `x` to double if it has real, negative components.
    Otherwise, output is just the array version of the input
    (via cupy.asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array : cupy.ndarray

    Examples
    --------
    >>> cupy.lib.scimath._fix_int_lt_zero([1,2])
    array([1, 2])
    >>> cupy.lib.scimath._fix_int_lt_zero([-1,2])
    array([-1.,  2.])

    See Also
    --------
    numpy.lib.scimath._fix_int_lt_zero

    """
    x = cupy.asarray(x)
    if cupy.any(cupy.isreal(x) & (x < 0)):
        x = x * 1.0
    return x


def _fix_real_abs_gt_1(x):
    """Convert `x` to complex if it has real components x_i with abs(x_i)>1.
    Otherwise, output is just the array version of the input
    (via cupy.asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array : cupy.ndarray

    Examples
    --------
    >>> cupy.lib.scimath._fix_real_abs_gt_1([0,1])
    array([0, 1])
    >>> cupy.lib.scimath._fix_real_abs_gt_1([0,2])
    array([0.+0.j, 2.+0.j])

    See Also
    --------
    numpy.lib.scimath._fix_real_abs_gt_1

    """
    x = cupy.asarray(x)
    if cupy.any(cupy.isreal(x) & (abs(x) > 1)):
        x = _tocomplex(x)
    return x


def sqrt(x):
    """
    Compute the square root of x.
    For negative input elements, a complex value is returned
    (unlike `cupy.sqrt` which returns NaN).

    Parameters
    ----------
    x : array_like
       The input value(s).

    Returns
    -------
    out : cupy.ndarray
       The square root of `x`.

    Examples
    --------
    For real, non-negative inputs this works just like `cupy.sqrt`:
    >>> cupy.emath.sqrt(1)
    array(1.)
    >>> cupy.emath.sqrt([1, 4])
    array([1., 2.])
    But it automatically handles negative inputs:
    >>> cupy.emath.sqrt(-1)
    array(0.+1.j)
    >>> cupy.emath.sqrt([-1, 4])
    array([0.+1.j, 2.+0.j])
    Different results are expected because:
    floating point 0.0 and -0.0 are distinct.
    For more control, explicitly use complex() as follows:
    >>> cupy.emath.sqrt(complex(-4.0, 0.0))
    array(0.+2.j)
    >>> cupy.emath.sqrt(complex(-4.0, -0.0))
    array(0.-2.j)

    See Also
    --------
    cupy.sqrt
    numpy.emath.sqrt

    """
    x = _fix_real_lt_zero(x)
    return cupy.sqrt(x)


def log(x):
    """
    Compute the natural logarithm of `x`.
    Return the "principal value" (for a description of this, see `cupy.log`)
    of :math:`log_e(x)`. For real `x > 0`, this is a real number (``log(0)``
    returns ``-inf`` and ``log(cupy.inf)`` returns ``inf``). Otherwise, the
    complex principle value is returned.

    Parameters
    ----------
    x : array_like
       The value(s) whose log is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The log of the `x` value(s).

    Notes
    -----
    For a log() that returns ``NAN`` when real `x < 0`, use `cupy.log`
    (note, however, that otherwise `cupy.log` and this `log` are identical,
    i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`, and,
    notably, the complex principle value if ``x.imag != 0``).

    Examples
    --------
    >>> cupy.emath.log(cupy.exp(1))
    array(1.)
    Negative arguments are handled "correctly" (recall that
    ``exp(log(x)) == x`` does *not* hold for real ``x < 0``):
    >>> cupy.emath.log(-cupy.exp(1)) == (1 + cupy.pi * 1j)
    array(True)

    See Also
    --------
    cupy.log
    numpy.emath.log

    """
    x = _fix_real_lt_zero(x)
    return cupy.log(x)


def log10(x):
    """
    Compute the logarithm base 10 of `x`.
    Return the "principal value" (for a description of this, see
    `cupy.log10`) of :math:`log_{10}(x)`. For real `x > 0`, this
    is a real number (``log10(0)`` returns ``-inf`` and ``log10(cupy.inf)``
    returns ``inf``). Otherwise, the complex principle value is returned.

    Parameters
    ----------
    x : array_like or scalar
       The value(s) whose log base 10 is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The log base 10 of the `x` value(s).

    Notes
    -----
    For a log10() that returns ``NAN`` when real `x < 0`, use `cupy.log10`
    (note, however, that otherwise `cupy.log10` and this `log10` are
    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,
    and, notably, the complex principle value if ``x.imag != 0``).

    Examples
    --------
    (We set the printing precision so the example can be auto-tested)
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.log10(10**1)
    array(1.)
    >>> cupy.emath.log10([-10**1, -10**2, 10**2])
    array([1.+1.3644j, 2.+1.3644j, 2.+0.j    ])

    See Also
    --------
    cupy.log10
    numpy.emath.log10

    """
    x = _fix_real_lt_zero(x)
    return cupy.log10(x)


def logn(n, x):
    """
    Take log base n of x.
    If `x` contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    n : array_like
       The integer base(s) in which the log is taken.
    x : array_like
       The value(s) whose log base `n` is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The log base `n` of the `x` value(s).

    Examples
    --------
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.logn(2, [4, 8])
    array([2., 3.])
    >>> cupy.emath.logn(2, [-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    See Also
    --------
    numpy.emath.logn

    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return cupy.log(x)/cupy.log(n)


def log2(x):
    """
    Compute the logarithm base 2 of `x`.
    Return the "principal value" (for a description of this, see
    `cupy.log2`) of :math:`log_2(x)`. For real `x > 0`, this is
    a real number (``log2(0)`` returns ``-inf`` and ``log2(cupy.inf)`` returns
    ``inf``). Otherwise, the complex principle value is returned.

    Parameters
    ----------
    x : array_like
       The value(s) whose log base 2 is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The log base 2 of the `x` value(s).

    Notes
    -----
    For a log2() that returns ``NAN`` when real `x < 0`, use `cupy.log2`
    (note, however, that otherwise `cupy.log2` and this `log2` are
    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,
    and, notably, the complex principle value if ``x.imag != 0``).

    Examples
    --------
    We set the printing precision so the example can be auto-tested:
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.log2(8)
    array(3.)
    >>> cupy.emath.log2([-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    See Also
    --------
    cupy.log2
    numpy.emath.log2

    """
    x = _fix_real_lt_zero(x)
    return cupy.log2(x)


def power(x, p):
    """
    Return x to the power p, (x**p).
    If `x` contains negative values, the output is converted to the
    complex domain.

    Parameters
    ----------
    x : array_like
        The input value(s).
    p : array_like of ints
        The power(s) to which `x` is raised. If `x` contains multiple values,
        `p` has to either be a scalar, or contain the same number of values
        as `x`. In the latter case, the result is
        ``x[0]**p[0], x[1]**p[1], ...``.

    Returns
    -------
    out : cupy.ndarray
        The result of ``x**p``.

    Examples
    --------
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.power([2, 4], 2)
    array([ 4, 16])
    >>> cupy.emath.power([2, 4], -2)
    array([0.25  ,  0.0625])
    >>> cupy.emath.power([-2, 4], 2)
    array([ 4.-0.j, 16.+0.j])

    See Also
    --------
    cupy.power
    numpy.emath.power

    """
    x = _fix_real_lt_zero(x)
    p = _fix_int_lt_zero(p)
    return cupy.power(x, p)


def arccos(x):
    """
    Compute the inverse cosine of x.
    Return the "principal value" (for a description of this, see
    `cupy.arccos`) of the inverse cosine of `x`. For real `x` such that
    `abs(x) <= 1`, this is a real number in the closed interval
    :math:`[0, \\pi]`.  Otherwise, the complex principle value is returned.

    Parameters
    ----------
    x : array_like or scalar
       The value(s) whose arccos is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The inverse cosine(s) of the `x` value(s).

    Notes
    -----
    For an arccos() that returns ``NAN`` when real `x` is not in the
    interval ``[-1,1]``, use `cupy.arccos`.

    Examples
    --------
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.arccos(1)
    array(0.)
    >>> cupy.emath.arccos([1, 2])
    array([0.-0.j   , 0.-1.317j])

    See Also
    --------
    cupy.arccos
    numpy.emath.arccos

    """
    x = _fix_real_abs_gt_1(x)
    return cupy.arccos(x)


def arcsin(x):
    """
    Compute the inverse sine of x.
    Return the "principal value" (for a description of this, see
    `cupy.arcsin`) of the inverse sine of `x`. For real `x` such that
    `abs(x) <= 1`, this is a real number in the closed interval
    :math:`[-\\pi/2, \\pi/2]`.  Otherwise, the complex principle value is
    returned.

    Parameters
    ----------
    x : array_like or scalar
       The value(s) whose arcsin is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The inverse sine(s) of the `x` value(s).

    Notes
    -----
    For an arcsin() that returns ``NAN`` when real `x` is not in the
    interval ``[-1,1]``, use `cupy.arcsin`.

    Examples
    --------
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.arcsin(0)
    array(0.)
    >>> cupy.emath.arcsin([0,1])
    array([0.    , 1.5708])

    See Also
    --------
    cupy.arcsin
    numpy.emath.arcsin

    """
    x = _fix_real_abs_gt_1(x)
    return cupy.arcsin(x)


def arctanh(x):
    """
    Compute the inverse hyperbolic tangent of `x`.
    Return the "principal value" (for a description of this, see
    `cupy.arctanh`) of ``arctanh(x)``. For real `x` such that
    ``abs(x) < 1``, this is a real number.  If `abs(x) > 1`, or if `x` is
    complex, the result is complex. Finally, `x = 1` returns``inf`` and
    ``x=-1`` returns ``-inf``.

    Parameters
    ----------
    x : array_like
       The value(s) whose arctanh is (are) required.

    Returns
    -------
    out : cupy.ndarray
       The inverse hyperbolic tangent(s) of the `x` value(s).

    Notes
    -----
    For an arctanh() that returns ``NAN`` when real `x` is not in the
    interval ``(-1,1)``, use `cupy.arctanh` (this latter, however, does
    return +/-inf for ``x = +/-1``).

    Examples
    --------
    >>> cupy.set_printoptions(precision=4)
    >>> cupy.emath.arctanh(cupy.eye(2))
    array([[inf,  0.],
           [ 0., inf]])
    >>> cupy.emath.arctanh([1j])
    array([0.+0.7854j])

    See Also
    --------
    cupy.arctanh
    numpy.emath.archtanh

    """
    x = _fix_real_abs_gt_1(x)
    return cupy.arctanh(x)

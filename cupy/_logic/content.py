from cupy import _core

import cupy


def _create_float_test_ufunc(name, doc):
    return _core.create_ufunc(
        'cupy_' + name,
        ('e->?', 'f->?', 'd->?', 'F->?', 'D->?',
         ), 'out0 = %s(in0)' % name,
        doc=doc)


isfinite = _create_float_test_ufunc(
    'isfinite',
    '''Tests finiteness elementwise.

    Each element of returned array is ``True`` only if the corresponding
    element of the input is finite (i.e. not an infinity nor NaN).

    .. seealso:: :data:`numpy.isfinite`

    ''')


isinf = _create_float_test_ufunc(
    'isinf',
    '''Tests if each element is the positive or negative infinity.

    .. seealso:: :data:`numpy.isinf`

    ''')


isnan = _create_float_test_ufunc(
    'isnan',
    '''Tests if each element is a NaN.

    .. seealso:: :data:`numpy.isnan`

    ''')


def isneginf(x, out=None):
    """Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : cupy.ndarray
        Input array.
    out : cupy.ndarray, optional
        A location into which the result is stored. If provided,
        it should have a shape that input broadcasts to.
        By default, None, a freshly- allocated boolean array,
        is returned.

    Returns
    -------
    y : cupy.ndarray
        Boolean array of same shape as ``x``.

    Examples
    --------
    >>> cupy.isneginf(0)
    array(False)
    >>> cupy.isneginf(-cupy.inf)
    array(True)
    >>> cupy.isneginf(cupy.array([-cupy.inf, -4, cupy.nan, 0, 4, cupy.inf]))
    array([ True, False, False, False, False, False])

    See Also
    --------
    numpy.isneginf

    """

    is_inf = isinf(x)
    try:
        signbit = cupy.signbit(x)
    except TypeError as e:
        dtype = x.dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e

    # TODO(khushi-411): Use `out` instead of `out=out` (see #6393)
    return cupy.logical_and(is_inf, signbit, out=out)


def isposinf(x, out=None):
    """Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : cupy.ndarray
        Input array.
    out : cupy.ndarray
        A location into which the result is stored. If provided,
        it should have a shape that input broadcasts to.
        By default, None, a freshly- allocated boolean array,
        is returned.

    Returns
    -------
    y : cupy.ndarray
        Boolean array of same shape as ``x``.

    Examples
    --------
    >>> cupy.isposinf(0)
    array(False)
    >>> cupy.isposinf(cupy.inf)
    array(True)
    >>> cupy.isposinf(cupy.array([-cupy.inf, -4, cupy.nan, 0, 4, cupy.inf]))
    array([False, False, False, False, False,  True])

    See Also
    --------
    numpy.isposinf

    """

    is_inf = isinf(x)
    try:
        signbit = ~cupy.signbit(x)
    except TypeError as e:
        dtype = x.dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e

    # TODO(khushi-411): Use `out` instead of `out=out` (see #6393)
    return cupy.logical_and(is_inf, signbit, out=out)

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
    """Returns a bool array, where True if input element is negative infinity.
    Args:
        x (cupy.ndarray): Input array.
    Returns:
        cupy.ndarray: Boolean array of same shape as ``x``.
    Examples
    --------
    >>> cupy.isneginf(0)
    False
    >>> cupy.isneginf([4, -4])
    [False, False]

    .. seealso:: :func:`numpy.isneginf`

    """

    is_inf = cupy.isinf(x)
    try:
        signbit = cupy.signbit(x)
    except TypeError as e:
        dtype = cupy.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                'as it would be uncertain.') from e
    else:
        return cupy.logical_and(is_inf, signbit, out=None)


def isposinf(x, out=None):
    """Returns a bool array, where True if input element is positive infinity.
    Args:
        x (cupy.ndarray): Input array.
    Returns:
        cupy.ndarray: Boolean array of same shape as ``x``.
    Examples
    --------
    >>> cupy.isposinf(0)
    False
    >>> cupy.isposinf([4, -4])
    [False, False]

    .. seealso:: :func:`numpy.isposinf`

    """

    is_inf = cupy.isinf(x)
    try:
        signbit = ~cupy.signbit(x)
    except TypeError as e:
        dtype = cupy.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'as it would be uncertain.') from e
    else:
        return cupy.logical_and(is_inf, signbit, out=None)

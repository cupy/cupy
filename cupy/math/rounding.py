from cupy import core
from cupy.core import fusion
from cupy.math import ufunc


@fusion._ufunc_wrapper(core.core._round_ufunc)
def around(a, decimals=0, out=None):
    """Rounds to the given number of decimals.

    Args:
        a (cupy.ndarray): The source array.
        decimals (int): umber of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of positions to
            the left of the decimal point.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Rounded array.

    .. seealso:: :func:`numpy.around`

    """
    a = core.array(a, copy=False)
    return a.round(decimals, out)


@fusion._ufunc_wrapper(core.core._round_ufunc)
def round_(a, decimals=0, out=None):
    a = core.array(a, copy=False)
    return a.round(decimals, out)


rint = ufunc.create_math_ufunc(
    'rint', 1, 'cupy_rint',
    '''Rounds each element of an array to the nearest integer.

    .. seealso:: :data:`numpy.rint`

    ''')


floor = ufunc.create_math_ufunc(
    'floor', 1, 'cupy_floor',
    '''Rounds each element of an array to its floor integer.

    .. seealso:: :data:`numpy.floor`

    ''', support_complex=False)


ceil = ufunc.create_math_ufunc(
    'ceil', 1, 'cupy_ceil',
    '''Rounds each element of an array to its ceiling integer.

    .. seealso:: :data:`numpy.ceil`

    ''', support_complex=False)


trunc = ufunc.create_math_ufunc(
    'trunc', 1, 'cupy_trunc',
    '''Rounds each element of an array towards zero.

    .. seealso:: :data:`numpy.trunc`

    ''', support_complex=False)


fix = core.create_ufunc(
    'cupy_fix', ('e->e', 'f->f', 'd->d'),
    'out0 = (in0 >= 0.0) ? floor(in0): ceil(in0)',
    doc='''If given value x is positive, it return floor(x).
    Else, it return ceil(x).

    .. seealso:: :func:`numpy.fix`

    ''')

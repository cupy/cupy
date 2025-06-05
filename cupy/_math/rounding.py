from __future__ import annotations

import warnings

from cupy import _core
from cupy._core import fusion
from cupy._math import ufunc


def around(a, decimals=0, out=None):
    """Rounds to the given number of decimals.

    Args:
        a (cupy.ndarray): The source array.
        decimals (int): Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of positions to
            the left of the decimal point.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Rounded array.

    .. seealso:: :func:`numpy.around`

    """
    if fusion._is_fusing():
        return fusion._call_ufunc(
            _core.core._round_ufunc, a, decimals, out=out)
    a = _core.array(a, copy=False)
    return a.round(decimals, out=out)


def round(a, decimals=0, out=None):
    return around(a, decimals, out=out)


def round_(a, decimals=0, out=None):
    warnings.warn('Please use `round` instead.', DeprecationWarning)
    return around(a, decimals, out=out)


rint = ufunc.create_math_ufunc(
    'rint', 1, 'cupy_rint',
    '''Rounds each element of an array to the nearest integer.

    .. seealso:: :data:`numpy.rint`

    ''')


def create_rounding_ufunc(name, op, doc):
    return _core.create_ufunc(
        name,
        (
            "?->?",
            "b->b",
            "B->B",
            "h->h",
            "H->H",
            "i->i",
            "I->I",
            "l->l",
            "L->L",
            "q->q",
            "Q->Q",
            ("e->e", op),
            ("f->f", op),
            ("d->d", op),
        ),
        "out0 = in0",
        doc=doc,
    )


floor = create_rounding_ufunc(
    "cupy_floor",
    "out0 = floor(in0)",
    """Rounds each element of an array to its floor integer.

    .. seealso:: :data:`numpy.floor`

    """,
)


ceil = create_rounding_ufunc(
    "cupy_ceil",
    "out0 = ceil(in0)",
    """Rounds each element of an array to its ceiling integer.

    .. seealso:: :data:`numpy.ceil`

    """,
)


trunc = create_rounding_ufunc(
    "cupy_trunc",
    "out0 = trunc(in0)",
    """Rounds each element of an array towards zero.

    .. seealso:: :data:`numpy.trunc`

    """,
)


fix = create_rounding_ufunc(
    "cupy_fix",
    "out0 = (in0 >= 0.0) ? floor(in0): ceil(in0)",
    """If given value x is positive, it return floor(x).
    Else, it return ceil(x).

    .. seealso:: :func:`numpy.fix`

    """,
)

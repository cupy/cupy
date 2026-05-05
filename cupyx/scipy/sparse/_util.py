from __future__ import annotations

import cupy
from cupy._core import core


def isdense(x):
    return isinstance(x, core.ndarray)


def isintlike(x):
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False


def isscalarlike(x):
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)


def isshape(x):
    """Return ``True`` if ``x`` is a 2-tuple of intlike values.

    Pure type check; does not reject negative dimensions.  Use
    :func:`check_shape` (or a follow-up explicit check) to enforce
    non-negativity with a scipy-compatible error message.
    """
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    m, n = x
    if isinstance(n, tuple):
        return False
    return isintlike(m) and isintlike(n)


def check_shape(shape):
    """Validate ``shape`` and return it canonicalized to ``(int, int)``.

    Accepts any 2-element iterable (tuple, list, etc.) of intlike
    values.  Raises ``ValueError`` with the same message scipy uses
    when the shape is not a 2-tuple of intlike values or contains a
    negative dimension.  Centralizing this here keeps the error
    message consistent across CuPy's sparse constructors.
    """
    try:
        shape_tuple = tuple(shape)
    except TypeError:
        raise ValueError('invalid shape (must be a 2-tuple of int)')
    if not isshape(shape_tuple):
        raise ValueError('invalid shape (must be a 2-tuple of int)')
    m, n = int(shape_tuple[0]), int(shape_tuple[1])
    if m < 0 or n < 0:
        raise ValueError("'shape' elements cannot be negative")
    return m, n

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


def isshape(x, allow_nd=(2,)):
    """Return ``True`` if ``x`` is a shape tuple of intlike values.

    Pure type check; does not reject negative dimensions.  Use
    :func:`check_shape` (or a follow-up explicit check) to enforce
    non-negativity with a scipy-compatible error message.

    Args:
        allow_nd (tuple of int): Tuple of accepted tuple lengths
            (number of dimensions).  Defaults to ``(2,)`` so 2-D-only
            callers are unaffected; pass e.g. ``(1, 2)`` to also accept
            1-D shapes.
    """
    if not isinstance(x, tuple) or len(x) not in allow_nd:
        return False
    # Reject nested tuples (e.g. ``((m, n),)``) and non-intlike entries.
    return all(not isinstance(v, tuple) and isintlike(v) for v in x)


def check_shape(shape, allow_nd=(2,)):
    """Validate ``shape`` and return it canonicalized to a tuple of ints.

    Accepts any iterable (tuple, list, etc.) of intlike values whose
    length is in ``allow_nd``.  Raises ``ValueError`` with the same
    message scipy uses when the shape is not a valid shape tuple or
    contains a negative dimension.

    Args:
        allow_nd (tuple of int): Accepted dimensionalities.  Defaults to
            ``(2,)``; pass ``(1, 2)`` to also accept 1-D shapes.
    """
    try:
        shape_tuple = tuple(shape)
    except TypeError:
        raise ValueError('invalid shape (must be a 2-tuple of int)')
    if not isshape(shape_tuple, allow_nd=allow_nd):
        raise ValueError('invalid shape (must be a 2-tuple of int)')
    out = tuple(int(v) for v in shape_tuple)
    if any(v < 0 for v in out):
        raise ValueError("'shape' elements cannot be negative")
    return out


def check_input_ndim(shape, input_ndim):
    """Reject an explicit ``shape`` that disagrees in dimensionality with a
    full array input.

    When ``arg1`` is a whole sparse/dense object its data already fixes the
    dimensionality; an explicit ``shape`` of a different ndim (e.g. a 1-D
    ``shape`` for a 2-D input) must not silently reinterpret the data --
    that corrupts the result.  ``shape`` of ``None`` (unspecified) is
    always accepted.
    """
    if shape is not None and len(shape) != input_ndim:
        raise ValueError(
            f'cannot use a {len(shape)}-D shape {tuple(shape)} with a '
            f'{input_ndim}-D input array')

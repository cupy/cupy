from __future__ import annotations

from cupy._core._memory_range import get_bound as _get_bounds


def byte_bounds(a):
    """Returns pointers to the end-points of an array.

    Args:
        a: ndarray
    Returns:
        Tuple[int, int]: pointers to the end-points of an array

    .. seealso:: :func:`numpy.byte_bounds`
    """
    return _get_bounds(a)

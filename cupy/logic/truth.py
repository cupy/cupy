import cupy
from cupy.core import _routines_logic as _logic
from cupy.core import fusion


def all(a, axis=None, out=None, keepdims=False):
    """Tests whether all array elements along a given axis evaluate to True.

    Args:
        a (cupy.ndarray): Input array.
        axis (int or tuple of ints): Along which axis to compute all.
            The flattened array is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: An array reduced of the input array along the axis.

    .. seealso:: :func:`numpy.all`

    """
    if fusion._is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.all does not support `keepdims` in fusion yet.')
        return fusion._call_reduction(
            _logic.all, a, axis=axis, out=out)

    assert isinstance(a, cupy.ndarray)
    return a.all(axis=axis, out=out, keepdims=keepdims)


def any(a, axis=None, out=None, keepdims=False):
    """Tests whether any array elements along a given axis evaluate to True.

    Args:
        a (cupy.ndarray): Input array.
        axis (int or tuple of ints): Along which axis to compute all.
            The flattened array is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: An array reduced of the input array along the axis.

    .. seealso:: :func:`numpy.any`

    """
    if fusion._is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.any does not support `keepdims` in fusion yet.')
        return fusion._call_reduction(
            _logic.any, a, axis=axis, out=out)

    assert isinstance(a, cupy.ndarray)
    return a.any(axis=axis, out=out, keepdims=keepdims)

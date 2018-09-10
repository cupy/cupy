import cupy
from cupy.core import core
from cupy.core import fusion


@fusion._reduction_wrapper(core._all)
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
    assert isinstance(a, cupy.ndarray)
    return a.all(axis=axis, out=out, keepdims=keepdims)


@fusion._reduction_wrapper(core._any)
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
    assert isinstance(a, cupy.ndarray)
    return a.any(axis=axis, out=out, keepdims=keepdims)

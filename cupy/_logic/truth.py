import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy import _util


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
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.all does not support `keepdims` in fusion yet.')
        return _fusion_thread_local.call_reduction(
            _logic.all, a, axis=axis, out=out)

    _util.check_array(a, arg_name='a')

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
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.any does not support `keepdims` in fusion yet.')
        return _fusion_thread_local.call_reduction(
            _logic.any, a, axis=axis, out=out)

    _util.check_array(a, arg_name='a')

    return a.any(axis=axis, out=out, keepdims=keepdims)


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Tests whether each element of a 1-D array is also present in a second
    array.

    Returns a boolean array the same length as ``ar1`` that is ``True``
    where an element of ``ar1`` is in ``ar2`` and ``False`` otherwise.

    Args:
        ar1 (cupy.ndarray): Input array.
        ar2 (cupy.ndarray): The values against which to test each value of
            ``ar1``.
        assume_unique (bool, optional): Ignored
        invert (bool, optional): If ``True``, the values in the returned array
            are inverted (that is, ``False`` where an element of ``ar1`` is in
            ``ar2`` and ``True`` otherwise). Default is ``False``.

    Returns:
        cupy.ndarray, bool: The values ``ar1[in1d]`` are in ``ar2``.

    """
    # Ravel both arrays, behavior for the first array could be different
    ar1 = ar1.ravel()
    ar2 = ar2.ravel()
    if ar1.size == 0 or ar2.size == 0:
        if invert:
            return cupy.ones(ar1.shape, dtype=cupy.bool_)
        else:
            return cupy.zeros(ar1.shape, dtype=cupy.bool_)
    # Use brilliant searchsorted trick
    # https://github.com/cupy/cupy/pull/4018#discussion_r495790724
    ar2 = cupy.sort(ar2)
    v1 = cupy.searchsorted(ar2, ar1, 'left')
    v2 = cupy.searchsorted(ar2, ar1, 'right')
    return v1 == v2 if invert else v1 != v2


def isin(element, test_elements, assume_unique=False, invert=False):
    """Calculates element in ``test_elements``, broadcasting over ``element``
    only. Returns a boolean array of the same shape as ``element`` that is
    ``True`` where an element of ``element`` is in ``test_elements`` and
    ``False`` otherwise.

    Args:
        element (cupy.ndarray): Input array.
        test_elements (cupy.ndarray): The values against which to test each
            value of ``element``. This argument is flattened if it is an
            array or array_like.
        assume_unique (bool, optional): Ignored
        invert (bool, optional): If ``True``, the values in the returned array
            are inverted, as if calculating element not in ``test_elements``.
            Default is ``False``.

    Returns:
        cupy.ndarray, bool:
            Has the same shape as ``element``. The values ``element[isin]``
            are in ``test_elements``.
    """
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)

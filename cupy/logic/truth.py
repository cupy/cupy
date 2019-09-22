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


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as ``ar1`` that is True
    where an element of ``ar1`` is in ``ar2`` and False otherwise.

    Args:
        ar1 (cupy.ndarray): Input array.
        ar2  (cupy.ndarray):
            The values against which to test each value of ``ar1``.
        assume_unique (bool, optional):
            If True, the input arrays are both assumed to be unique, which
            can speed up the calculation. Default is False.
        invert(bool, optional):
            If True, the values in the returned array are inverted (that is,
            False where an element of ``ar1`` is in ``ar2`` and True otherwise)
            Default is False.

    Returns:
        cupy.ndarray, bool: The values ``ar1[in1d]`` are in ``ar2``.

    """
    # TODO(UmashankarTriforce): Improve efficiency of len check
    # Ravel both arrays, behavior for the first array could be different
    # check for shape
    if not(ar1.shape[-1]):
        return cupy.array([], dtype=bool)
    ar1 = cupy.asarray(ar1).ravel()
    ar2 = cupy.asarray(ar2).ravel()

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = cupy.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = cupy.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = cupy.unique(ar1, return_inverse=True)
        ar2 = cupy.unique(ar2)

    ar = cupy.concatenate((ar1, ar2))

    order = ar.argsort()
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = cupy.concatenate((bool_ar, cupy.array(invert)))
    ret = cupy.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False):
    """
    Calculates ``element in test_elements``, broadcasting over ``element`` only
    Returns a boolean array of the same shape as ``element`` that is True
    where an element of ``element`` is in ``test_elements``
    and False otherwise.


    Args:
        element (cupy.ndarray): Input array.
        test_elements (cupy.ndarray):
            The values against which to test each value of `element`.
            This argument is flattened if it is an array or array_like.
            See notes for behavior with non-array-like parameters.
        assume_unique (bool, optional):
            If True, the input arrays are both assumed to be unique, which
            can speed up the calculation.  Default is False.
        invert (bool, optional):
            If True, the values in the returned array are inverted, as if
            calculating `element not in test_elements`. Default is False.

    Returns:
        cupy.ndarray, bool:
            Has the same shape as `element`. The values `element[isin]`
            are in `test_elements`.

    """
    element = cupy.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)

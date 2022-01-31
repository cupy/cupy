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


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Args:
        ar1 (cupy.ndarray): Input array.
        ar2 (cupy.ndarray): The values against which to test each value of
            ``ar1``.
        assume_unique (bool, optional): If ``True``, the input arrays are
            both assumed to be unique, which can speed up the calculation.
            If ``True`` but ``ar1`` or ``ar2`` are not unique, incorrect
            results and out-of-bounds indices could result.
            Default is ``False``.
        return_indices (bool, optional): If ``True``, the indices which
            correspond to the intersection of the two arrays are returned.
            The first instance of a value is used if there are multiple.
            Default is ``False``.

    Returns:
        intersect1d (cupy.ndarray):
            Sorted 1D array of common and unique elements.

        comm1 (cupy.ndarray):
            The indices of the first occurrences of the common values
            in ``ar1``. Only provided if ``return_indices`` is True.

        comm2 (cupy.ndarray):
            The indices of the first occurrences of the common values
            in ``ar2``. Only provided if ``return_indices`` is True.

    .. seealso:: :func:`numpy.intersect1d`

    """

    if not assume_unique:
        if return_indices:
            ar1, ind1 = cupy.unique(ar1, return_index=True)
            ar2, ind2 = cupy.unique(ar2, return_index=True)
        else:
            ar1 = cupy.unique(ar1)
            ar2 = cupy.unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    if return_indices:
        ar2_sort_indices = cupy.argsort(ar2)
        ar2 = ar2[ar2_sort_indices]
    else:
        ar2 = cupy.sort(ar2)

    # Use brilliant searchsorted trick
    # https://github.com/cupy/cupy/pull/4018#discussion_r495790724
    v1 = cupy.searchsorted(ar2, ar1, 'left')
    v2 = cupy.searchsorted(ar2, ar1, 'right')

    mask = v1 != v2

    int1d = ar1[mask]

    if return_indices:
        ar1_indices = cupy.flatnonzero(mask)
        ar2_indices = ar2_sort_indices[v2[mask]-1]
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d


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

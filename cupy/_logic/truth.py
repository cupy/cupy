import cupy
from cupy._core import _routines_logic as _logic
from cupy._core import _fusion_thread_local
from cupy._sorting import search as _search
from cupy import _util


_setxorkernel = cupy._core.ElementwiseKernel(
    'raw T X, int64 len',
    'bool z',
    'z = (i == 0 || X[i] != X[i-1]) && (i == len - 1 || X[i] != X[i+1])',
    'setxorkernel'
)


def all(a, axis=None, out=None, keepdims=False):
    """Tests whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : cupy.ndarray
        Input array.
    axis : int or tuple of ints
        Along which axis to compute all.
        The flattened array is used by default.
    out : cupy.ndarray
        Output array.
    keepdims : bool
        If ``True``, the axis is remained as an axis of size one.

    Returns
    -------
    y : cupy.ndarray
        An array reduced of the input array along the axis.

    See Also
    --------
    numpy.all

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

    Parameters
    ----------
    a : cupy.ndarray
        Input array.
    axis : int or tuple of ints
        Along which axis to compute all.
        The flattened array is used by default.
    out : cupy.ndarray
        Output array.
    keepdims : bool
        If ``True``, the axis is remained as an axis of size one.

    Returns
    -------
    y : cupy.ndarray
        An array reduced of the input array along the axis.

    See Also
    --------
    numpy.any

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

    Parameters
    ----------
    ar1 : cupy.ndarray
        Input array.
    ar2 : cupy.ndarray
        The values against which to test each value of ``ar1``.
    assume_unique : bool, optional
        Ignored
    invert : bool, optional
        If ``True``, the values in the returned array
        are inverted (that is, ``False`` where an element of ``ar1`` is in
        ``ar2`` and ``True`` otherwise). Default is ``False``.

    Returns
    -------
    y : cupy.ndarray, bool
        The values ``ar1[in1d]`` are in ``ar2``.

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
    return _search._exists_kernel(ar1, ar2, ar2.size, invert)


def intersect1d(arr1, arr2, assume_unique=False, return_indices=False):
    """Find the intersection of two arrays.
    Returns the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    arr1, arr2 : cupy.ndarray
        Input arrays. Arrays will be flattened if they are not in 1D.
    assume_unique : bool
        By default, False. If set True, the input arrays will be
        assumend to be unique, which speeds up the calculation. If set True,
        but the arrays are not unique, incorrect results and out-of-bounds
        indices could result.
    return_indices : bool
       By default, False. If True, the indices which correspond to the
       intersection of the two arrays are returned.

    Returns
    -------
    intersect1d : cupy.ndarray
        Sorted 1D array of common and unique elements.
    comm1 : cupy.ndarray
        The indices of the first occurrences of the common values
        in `arr1`. Only provided if `return_indices` is True.
    comm2 : cupy.ndarray
        The indices of the first occurrences of the common values
        in `arr2`. Only provided if `return_indices` is True.

    See Also
    --------
    numpy.intersect1d

    """
    if not assume_unique:
        if return_indices:
            arr1, ind1 = cupy.unique(arr1, return_index=True)
            arr2, ind2 = cupy.unique(arr2, return_index=True)
        else:
            arr1 = cupy.unique(arr1)
            arr2 = cupy.unique(arr2)
    else:
        arr1 = arr1.ravel()
        arr2 = arr2.ravel()

    if not return_indices:
        mask = _search._exists_kernel(arr1, arr2, arr2.size, False)
        return arr1[mask]

    mask, v1 = _search._exists_and_searchsorted_kernel(
        arr1, arr2, arr2.size, False)
    int1d = arr1[mask]
    arr1_indices = cupy.flatnonzero(mask)
    arr2_indices = v1[mask]

    if not assume_unique:
        arr1_indices = ind1[arr1_indices]
        arr2_indices = ind2[arr2_indices]

    return int1d, arr1_indices, arr2_indices


def isin(element, test_elements, assume_unique=False, invert=False):
    """Calculates element in ``test_elements``, broadcasting over ``element``
    only. Returns a boolean array of the same shape as ``element`` that is
    ``True`` where an element of ``element`` is in ``test_elements`` and
    ``False`` otherwise.

    Parameters
    ----------
    element : cupy.ndarray
        Input array.
    test_elements : cupy.ndarray
        The values against which to test each
        value of ``element``. This argument is flattened if it is an
        array or array_like.
    assume_unique : bool, optional
        Ignored
    invert : bool, optional
        If ``True``, the values in the returned array
        are inverted, as if calculating element not in ``test_elements``.
        Default is ``False``.

    Returns
    -------
    y : cupy.ndarray, bool
        Has the same shape as ``element``. The values ``element[isin]``
        are in ``test_elements``.

    """
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)


def setdiff1d(ar1, ar2, assume_unique=False):
    """Find the set difference of two arrays. It returns unique
    values in `ar1` that are not in `ar2`.

    Parameters
    ----------
    ar1 : cupy.ndarray
        Input array
    ar2 : cupy.ndarray
        Input array for comparision
    assume_unique : bool
        By default, False, i.e. input arrays are not unique.
        If True, input arrays are assumed to be unique. This can
        speed up the calculation.

    Returns
    -------
    setdiff1d : cupy.ndarray
        Returns a 1D array of values in `ar1` that are not in `ar2`.
        It always returns a sorted output for unsorted input only
        if `assume_unique=False`.

    See Also
    --------
    numpy.setdiff1d

    """
    if assume_unique:
        ar1 = cupy.ravel(ar1)
    else:
        ar1 = cupy.unique(ar1)
        ar2 = cupy.unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]


def setxor1d(ar1, ar2, assume_unique=False):
    """Find the set exclusive-or of two arrays.

    Parameters
    ----------
    ar1, ar2 : cupy.ndarray
        Input arrays. They are flattend if they are not already 1-D.
    assume_unique : bool
        By default, False, i.e. input arrays are not unique.
        If True, input arrays are assumed to be unique. This can
        speed up the calculation.

    Returns
    -------
    setxor1d : cupy.ndarray
        Return the sorted, unique values that are in only one
        (not both) of the input arrays.

    See Also
    --------
    numpy.setxor1d

    """
    if not assume_unique:
        ar1 = cupy.unique(ar1)
        ar2 = cupy.unique(ar2)

    aux = cupy.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux

    aux.sort()

    return aux[_setxorkernel(aux, aux.size,
                             cupy.zeros(aux.size, dtype=cupy.bool_))]


def union1d(arr1, arr2):
    """Find the union of two arrays.

    Returns the unique, sorted array of values that are in either of
    the two input arrays.

    Parameters
    ----------
    arr1, arr2 : cupy.ndarray
        Input arrays. They are flattend if they are not already 1-D.

    Returns
    -------
    union1d : cupy.ndarray
        Sorted union of the input arrays.

    See Also
    --------
    numpy.union1d

    """
    return cupy.unique(cupy.concatenate((arr1, arr2), axis=None))

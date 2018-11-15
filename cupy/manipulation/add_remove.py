import numpy

import cupy
from cupy import core


# TODO(okuta): Implement delete


# TODO(okuta): Implement insert


def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Args:
        arr(array_like):
            Values are appended to a copy of this array.
        values(array_like):
            These values are appended to a copy of `arr`.  It must be of the
            correct shape (the same shape as `arr`, excluding `axis`).  If
            `axis` is not specified, `values` can be any shape and will be
            flattened before use.
        axis(int, optional):
            The axis along which `values` are appended.  If `axis` is not
            given, both `arr` and `values` are flattened before use.

    Returns:
        ndarray
            A copy of `arr` with `values` appended to `axis`.  Note that
            `append` does not occur in-place: a new array is allocated and
            filled.  If `axis` is None, `out` is a flattened array.

    .. seealso:: :func:`numpy.append`
    """
    arr = cupy.asarray(arr)
    values = cupy.asarray(values)
    if axis is None:
        return core.concatenate_method(
            (arr.ravel(), values.ravel()), 0).ravel()
    return core.concatenate_method((arr, values), axis)


def resize(a, new_shape):
    """Return a new array with the specified shape.

    If the new array is larger than the original array, then the new
    array is filled with repeated copies of `a`.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of `a`.

    Args:
        a(array_like): Array to be resized.
        new_shape(int or tuple of int): Shape of resized array.

    Returns:
        ndarray:
            The new array is formed from the data in the old array, repeated
            if necessary to fill out the required number of elements.  The
            data are repeated in the order that they are stored in memory.

    .. seealso:: :func:`numpy.resize`
    """
    if numpy.isscalar(a):
        return cupy.full(new_shape, a)
    if a.shape == () or len(a) == 0:
        # 0-dim or empty array
        return cupy.zeros(new_shape, dtype=a.dtype)
    a = a.ravel()
    reps, remainder = divmod(numpy.prod(new_shape), len(a))
    return core.concatenate_method(
        [a] * reps + [a[:remainder]], 0).reshape(new_shape)


# TODO(okuta): Implement trim_zeros


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    """Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Args:
        ar(array_like): Input array. This will be flattened if it is not
            already 1-D.
        return_index(bool, optional): If True, also return the indices of `ar`
            (along the specified axis, if provided, or in the flattened array)
            that result in the unique array.
        return_inverse(bool, optional): If True, also return the indices of the
            unique array (for the specified axis, if provided) that can be used
            to reconstruct `ar`.
        return_counts(bool, optional): If True, also return the number of times
            each unique item appears in `ar`.
        axis(int or None, optional): Not supported yet.

    Returns:
        cupy.ndarray or tuple:
            If there are no optional outputs, it returns the
            :class:`cupy.ndarray` of the sorted unique values. Otherwise, it
            returns the tuple which contains the sorted unique values and
            followings.

            * The indices of the first occurrences of the unique values in the
              original array. Only provided if `return_index` is True.
            * The indices to reconstruct the original array from the
              unique array. Only provided if `return_inverse` is True.
            * The number of times each of the unique values comes up in the
              original array. Only provided if `return_counts` is True.

    .. seealso:: :func:`numpy.unique`
    """
    if axis is not None:
        raise NotImplementedError('axis option is not supported yet.')

    ar = cupy.asarray(ar).flatten()

    if return_index or return_inverse:
        perm = ar.argsort()
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = cupy.empty(aux.shape, dtype=cupy.bool_)
    mask[0] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = aux[mask]
    if not return_index and not return_inverse and not return_counts:
        return ret

    ret = ret,
    if return_index:
        ret += perm[mask],
    if return_inverse:
        imask = cupy.cumsum(mask) - 1
        inv_idx = cupy.empty(mask.shape, dtype=cupy.intp)
        inv_idx[perm] = imask
        ret += inv_idx,
    if return_counts:
        nonzero = cupy.nonzero(mask)[0]
        idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],
    return ret

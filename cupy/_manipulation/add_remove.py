import numpy

import cupy
import math
from cupy import _core


def delete(arr, indices, axis=None):
    """
    Delete values from an array along the specified axis.

    Args:
        arr (cupy.ndarray):
            Values are deleted from a copy of this array.
        indices (slice, int or array of ints):
            These indices correspond to values that will be deleted from the
            copy of `arr`.
            Boolean indices are treated as a mask of elements to remove.
        axis (int or None):
            The axis along which `indices` correspond to values that will be
            deleted. If `axis` is not given, `arr` will be flattened.

    Returns:
        cupy.ndarray:
            A copy of `arr` with values specified by `indices` deleted along
            `axis`.

    .. warning:: This function may synchronize the device.

    .. seealso:: :func:`numpy.delete`.
    """

    if axis is None:

        arr = arr.ravel()

        if isinstance(indices, cupy.ndarray) and indices.dtype == cupy.bool_:
            return arr[~indices]

        mask = cupy.ones(arr.size, dtype=bool)
        mask[indices] = False
        return arr[mask]

    else:

        if isinstance(indices, cupy.ndarray) and indices.dtype == cupy.bool_:
            return cupy.compress(~indices, arr, axis=axis)

        mask = cupy.ones(arr.shape[axis], dtype=bool)
        mask[indices] = False
        return cupy.compress(mask, arr, axis=axis)


# TODO(okuta): Implement insert


def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Args:
        arr (array_like):
            Values are appended to a copy of this array.
        values (array_like):
            These values are appended to a copy of ``arr``.  It must be of the
            correct shape (the same shape as ``arr``, excluding ``axis``).  If
            ``axis`` is not specified, ``values`` can be any shape and will be
            flattened before use.
        axis (int or None):
            The axis along which ``values`` are appended.  If ``axis`` is not
            given, both ``arr`` and ``values`` are flattened before use.

    Returns:
        cupy.ndarray:
            A copy of ``arr`` with ``values`` appended to ``axis``.  Note that
            ``append`` does not occur in-place: a new array is allocated and
            filled.  If ``axis`` is None, ``out`` is a flattened array.

    .. seealso:: :func:`numpy.append`
    """
    # TODO(asi1024): Implement fast path for scalar inputs.
    arr = cupy.asarray(arr)
    values = cupy.asarray(values)
    if axis is None:
        return _core.concatenate_method(
            (arr.ravel(), values.ravel()), 0).ravel()
    return _core.concatenate_method((arr, values), axis)


_resize_kernel = _core.ElementwiseKernel(
    'raw T x, int64 size', 'T y',
    'y = x[i % size]',
    'cupy_resize',
)


def resize(a, new_shape):
    """Return a new array with the specified shape.

    If the new array is larger than the original array, then the new
    array is filled with repeated copies of ``a``.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of ``a``.

    Args:
        a (array_like): Array to be resized.
        new_shape (int or tuple of int): Shape of resized array.

    Returns:
        cupy.ndarray:
            The new array is formed from the data in the old array, repeated
            if necessary to fill out the required number of elements.  The
            data are repeated in the order that they are stored in memory.

    .. seealso:: :func:`numpy.resize`
    """
    if numpy.isscalar(a):
        return cupy.full(new_shape, a)
    a = cupy.asarray(a)
    if a.size == 0:
        return cupy.zeros(new_shape, dtype=a.dtype)
    out = cupy.empty(new_shape, a.dtype)
    _resize_kernel(a, a.size, out)
    return out


_first_nonzero_krnl = _core.ReductionKernel(
    'T data, int64 len',
    'int64 y',
    'data == T(0) ? len : _j',
    'min(a, b)',
    'y = a',
    'len',
    'first_nonzero'
)


def trim_zeros(filt, trim='fb'):
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Returns the trimmed array

    Args:
        filt(cupy.ndarray): Input array
        trim(str, optional):
            'fb' default option trims the array from both sides.
            'f' option trim zeros from front.
            'b' option trim zeros from back.

    Returns:
        cupy.ndarray: trimmed input

    .. seealso:: :func:`numpy.trim_zeros`

    """
    if filt.ndim > 1:
        raise ValueError('Multi-dimensional trim is not supported')
    if not filt.ndim:
        raise TypeError('0-d array cannot be trimmed')
    start = 0
    end = filt.size
    trim = trim.upper()
    if 'F' in trim:
        start = _first_nonzero_krnl(filt, filt.size).item()
    if 'B' in trim:
        end = filt.size - _first_nonzero_krnl(filt[::-1], filt.size).item()
    return filt[start:end]


@_core.fusion.fuse()
def _unique_update_mask_equal_nan(mask, x0):
    mask1 = cupy.logical_not(cupy.isnan(x0))
    mask[:] = cupy.logical_and(mask, mask1)


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None, *, equal_nan=True):
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
        axis(int or None, optional): The axis to operate on. If None, ar will
            be flattened. If an integer, the subarrays indexed by the given
            axis will be flattened and treated as the elements of a 1-D array
            with the dimension of the given axis, see the notes for more
            details. The default is None.
        equal_nan(bool, optional): If True, collapse multiple NaN values in the
            return array into one.

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

    Notes:
       When an axis is specified the subarrays indexed by the axis are sorted.
       This is done by making the specified axis the first dimension of the
       array (move the axis to the first dimension to keep the order of the
       other axes) and then flattening the subarrays in C order.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.unique`
    """
    if axis is None:
        ret = _unique_1d(ar, return_index=return_index,
                         return_inverse=return_inverse,
                         return_counts=return_counts,
                         equal_nan=equal_nan)
        return ret

    ar = cupy.moveaxis(ar, axis, 0)

    # The array is reshaped into a contiguous 2D array
    orig_shape = ar.shape
    idx = cupy.arange(0, orig_shape[0], dtype=cupy.intp)
    ar = ar.reshape(orig_shape[0], math.prod(orig_shape[1:]))
    ar = cupy.ascontiguousarray(ar)
    is_unsigned = cupy.issubdtype(ar.dtype, cupy.unsignedinteger)
    is_complex = cupy.iscomplexobj(ar)

    ar_cmp = ar
    if is_unsigned:
        ar_cmp = ar.astype(cupy.intp)

    def compare_axis_elems(idx1, idx2):
        left, right = ar_cmp[idx1], ar_cmp[idx2]
        comp = cupy.trim_zeros(left - right, 'f')
        if comp.shape[0] > 0:
            diff = comp[0]
            if is_complex and cupy.isnan(diff):
                return True
            return diff < 0
        return False

    # The array is sorted lexicographically using the first item of each
    # element on the axis
    sorted_indices = cupy.empty(orig_shape[0], dtype=cupy.intp)
    queue = [(idx.tolist(), 0)]
    while queue != []:
        current, off = queue.pop(0)
        if current == []:
            continue

        mid_elem = current[0]
        left = []
        right = []
        for i in range(1, len(current)):
            if compare_axis_elems(current[i], mid_elem):
                left.append(current[i])
            else:
                right.append(current[i])

        elem_pos = off + len(left)
        queue.append((left, off))
        queue.append((right, elem_pos + 1))

        sorted_indices[elem_pos] = mid_elem

    ar = ar[sorted_indices]

    if ar.size > 0:
        mask = cupy.empty(ar.shape, dtype=cupy.bool_)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]

        mask = cupy.any(mask, axis=1)
    else:
        # If empty, then the mask should grab the first empty array as the
        # unique one
        mask = cupy.ones((ar.shape[0]), dtype=cupy.bool_)
        mask[1:] = False

    # Index the input array with the unique elements and reshape it into the
    # original size and dimension order
    ar = ar[mask]
    ar = ar.reshape(mask.sum().item(), *orig_shape[1:])
    ar = cupy.moveaxis(ar, 0, axis)

    ret = ar,
    if return_index:
        ret += sorted_indices[mask],
    if return_inverse:
        imask = cupy.cumsum(mask) - 1
        inv_idx = cupy.empty(mask.shape, dtype=cupy.intp)
        inv_idx[sorted_indices] = imask
        ret += inv_idx,
    if return_counts:
        nonzero = cupy.nonzero(mask)[0]  # may synchronize
        idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],

    if len(ret) == 1:
        ret = ret[0]
    return ret


def _unique_1d(ar, return_index=False, return_inverse=False,
               return_counts=False, equal_nan=True):
    ar = cupy.asarray(ar).flatten()

    if return_index or return_inverse:
        perm = ar.argsort()
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = cupy.empty(aux.shape, dtype=cupy.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    if equal_nan:
        _unique_update_mask_equal_nan(mask[1:], aux[:-1])

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
        nonzero = cupy.nonzero(mask)[0]  # may synchronize
        idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],
    return ret

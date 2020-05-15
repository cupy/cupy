import cupy

from cupy import core

# TODO(okuta): Implement delete


# TODO(okuta): Implement insert


# TODO(okuta): Implement append


# TODO(okuta): Implement resize

_first_nonzero_krnl = core.ReductionKernel(
    'raw S data, T indices, T len',
    'T y',
    'data[_j] ? _j : len',
    'min(a, b)',
    'y = a',
    'len',
    'first_nonzero'
)
_first_nonzero_cplx_krnl = core.ReductionKernel(
    'raw S real, raw S imag, T indices, T len',
    'T y',
    'real[_j] || imag[_j] ? _j : len',
    'min(a,b)',
    'y = a',
    'len',
    'first_nonzero_cplx'
)


def trim_zeros(filt, trim='fb'):
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Returns the trimmed array

    Args:
        filt(1D array or sequence): Input array
        trim(str, optional):
            'fb' default option trims the array from both sides.
            'f' option trim zeros from front.
            'b' option trim zeros from front

    Returns:
        cupy.ndarray: trimmed input

    .. seealso:: :func:`numpy.trim_zeros`
    """
    if filt.ndim > 1:
        raise ValueError("Multi-dimensional trim is not supported")
    start, end = cupy.asarray([0, filt.size - 1], dtype=int)
    indices = cupy.arange(filt.size)
    trim = trim.upper()
    if 'F' in trim:
        if filt.dtype.kind == 'c':
            start = _first_nonzero_cplx_krnl(
                filt.real, filt.imag, indices, filt.size)
        else:
            start = _first_nonzero_krnl(filt, indices, filt.size)
    if 'B' in trim:
        if filt.dtype.kind == 'c':
            end = filt.size - _first_nonzero_cplx_krnl(
                filt[::-1].real, filt[::-1].imag, indices, filt.size) - 1
        else:
            end = filt.size - _first_nonzero_krnl(
                filt[::-1], indices, filt.size) - 1
    return filt[start:end + 1]


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

    .. warning::

        This function may synchronize the device.

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
        nonzero = cupy.nonzero(mask)[0]  # may synchronize
        idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],
    return ret

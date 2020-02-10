import cupy
from cupy.core import _errors


def take(a, indices, axis=None, out=None):
    """Takes elements of an array at specified indices along an axis.

    This is an implementation of "fancy indexing" at single axis.

    This function does not support ``mode`` option.

    Args:
        a (cupy.ndarray): Array to extract elements.
        indices (int or array-like): Indices of elements that this function
            takes.
        axis (int): The axis along which to select indices. The flattened input
            is used by default.
        out (cupy.ndarray): Output array. If provided, it should be of
            appropriate shape and dtype.

    Returns:
        cupy.ndarray: The result of fancy indexing.

    .. seealso:: :func:`numpy.take`

    """
    # TODO(okuta): check type
    return a.take(indices, axis, out)


def take_along_axis(a, indices, axis):
    """Take values from the input array by matching 1d index and data slices.

    Args:
        a (cupy.ndarray): Array to extract elements.
        indices (cupy.ndarray): Indices to take along each 1d slice of ``a``.
        axis (int): The axis to take 1d slices along.

    Returns:
        cupy.ndarray: The indexed result.

    .. seealso:: :func:`numpy.take_along_axis`
    """

    if indices.dtype.kind not in ('i', 'u'):
        raise IndexError('`indices` must be an integer array')

    if axis is None:
        a = a.ravel()
        axis = 0

    ndim = a.ndim

    if not (-ndim <= axis < ndim):
        raise _errors._AxisError('Axis overrun')

    axis %= a.ndim

    if ndim != indices.ndim:
        raise ValueError(
            '`indices` and `a` must have the same number of dimensions')

    fancy_index = []
    for i, n in enumerate(a.shape):
        if i == axis:
            fancy_index.append(indices)
        else:
            ind_shape = (1,) * i + (-1,) + (1,) * (ndim - i - 1)
            fancy_index.append(cupy.arange(n).reshape(ind_shape))

    return a[fancy_index]


def choose(a, choices, out=None, mode='raise'):
    return a.choose(choices, out, mode)


# TODO(okuta): Implement compress


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Returns specified diagonals.

    This function extracts the diagonals along two specified axes. The other
    axes are not changed. This function returns a writable view of this array
    as NumPy 1.10 will do.

    Args:
        a (cupy.ndarray): Array from which the diagonals are taken.
        offset (int): Index of the diagonals. Zero indicates the main
            diagonals, a positive value upper diagonals, and a negative value
            lower diagonals.
        axis1 (int): The first axis to take diagonals from.
        axis2 (int): The second axis to take diagonals from.

    Returns:
        cupy.ndarray: A view of the diagonals of ``a``.

    .. seealso:: :func:`numpy.diagonal`

    """
    # TODO(okuta): check type
    return a.diagonal(offset, axis1, axis2)


# TODO(okuta): Implement select

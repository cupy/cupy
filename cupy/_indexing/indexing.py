import cupy
from cupy._core import internal


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

    axis = internal._normalize_axis_index(axis, ndim)

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

    return a[tuple(fancy_index)]


def choose(a, choices, out=None, mode='raise'):
    return a.choose(choices, out, mode)


def compress(condition, a, axis=None, out=None):
    """Returns selected slices of an array along given axis.

    Args:
        condition (1-D array of bools): Array that selects which entries to
            return. If len(condition) is less than the size of a along the
            given axis, then output is truncated to the length of the condition
            array.
        a (cupy.ndarray): Array from which to extract a part.
        axis (int): Axis along which to take slices. If None (default), work
            on the flattened array.
        out (cupy.ndarray): Output array. If provided, it should be of
            appropriate shape and dtype.

    Returns:
        cupy.ndarray: A copy of a without the slices along axis for which
            condition is false.

    .. warning::

            This function may synchronize the device.


    .. seealso:: :func:`numpy.compress`

    """
    return a.compress(condition, axis, out)


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


def extract(condition, a):
    """Return the elements of an array that satisfy some condition.

    This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.
    If ``condition`` is boolean, ``np.extract`` is equivalent to
    ``arr[condition]``.

    Args:
        condition (int or array_like): An array whose nonzero or True entries
            indicate the elements of array to extract.
        a (cupy.ndarray): Input array of the same size as condition.

    Returns:
        cupy.ndarray: Rank 1 array of values from arr where condition is True.

    .. warning::

            This function may synchronize the device.

    .. seealso:: :func:`numpy.extract`
    """

    if not isinstance(a, cupy.ndarray):
        raise TypeError('extract requires input array to be cupy.ndarray')

    if not isinstance(condition, cupy.ndarray):
        condition = cupy.array(condition)

    a = a.ravel()
    condition = condition.ravel()

    return a.take(condition.nonzero()[0])


def select(condlist, choicelist, default=0):
    """Return an array drawn from elements in choicelist, depending on conditions.

    Args:
        condlist (list of bool arrays): The list of conditions which determine
            from which array in `choicelist` the output elements are taken.
            When multiple conditions are satisfied, the first one encountered
            in `condlist` is used.
        choicelist (list of cupy.ndarray): The list of arrays from which the
            output elements are taken. It has to be of the same length
            as `condlist`.
        default (scalar) : If provided, will fill element inserted in `output`
            when all conditions evaluate to False. default value is 0.

    Returns:
        cupy.ndarray: The output at position m is the m-th element of the
        array in `choicelist` where the m-th element of the corresponding
        array in `condlist` is True.

    .. seealso:: :func:`numpy.select`
    """

    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    if not cupy.isscalar(default):
        raise TypeError("default only accepts scalar values")

    for i in range(len(choicelist)):
        if not isinstance(choicelist[i], cupy.ndarray):
            raise TypeError("choicelist only accepts lists of cupy ndarrays")
        cond = condlist[i]
        if cond.dtype.type is not cupy.bool_:
            raise ValueError(
                'invalid entry {} in condlist: should be boolean ndarray'
                .format(i))

    dtype = cupy.result_type(*choicelist)

    condlist = cupy.broadcast_arrays(*condlist)
    choicelist = cupy.broadcast_arrays(*choicelist, default)

    if choicelist[0].ndim == 0:
        result_shape = condlist[0].shape
    else:
        result_shape = cupy.broadcast_arrays(condlist[0],
                                             choicelist[0])[0].shape

    result = cupy.empty(result_shape, dtype)
    cupy.copyto(result, default)

    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        cupy.copyto(result, choice, where=cond)

    return result

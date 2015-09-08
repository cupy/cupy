import collections

from cupy import internal


def reshape(a, newshape):
    """Returns an array with new shape and same elements.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support ``order`` option.

    Args:
        a (cupy.ndarray): Array to be reshaped.
        newshape (int or tuple of ints): The new shape of the array to return.
            If it is an integer, then it is treated as a tuple of length one.
            It should be compatible with ``a.size``. One of the elements can be
            -1, which is automatically replaced with the appropriate value to
            make the shape compatible with ``a.size``.

    Returns:
        cupy.ndarray: A reshaped view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.reshape`

    """
    # TODO(beam2d): Support ordering option
    if isinstance(newshape, collections.Sequence):
        newshape = tuple(newshape)
    else:
        newshape = newshape,

    shape = a.shape
    if newshape == shape:
        return a.view()

    size = a.size
    newshape = internal.infer_unknown_dimension(newshape, size)
    if newshape == shape:
        return a.view()
    if internal.prod(newshape) != size:
        raise RuntimeError('Total size mismatch on reshape')

    newstrides = internal.get_strides_for_nocopy_reshape(a, newshape)
    if newstrides is not None:
        newarray = a.view()
    else:
        newarray = a.copy()
        newstrides = internal.get_strides_for_nocopy_reshape(
            newarray, newshape)
    newarray._shape = newshape
    newarray._strides = newstrides
    if newarray._c_contiguous == 1:
        newarray._f_contiguous = int(
            not size or len(shape) - shape.count(1) <= 1)
    else:
        newarray._f_contiguous = -1
    return newarray


def ravel(a):
    """Returns a flattened array.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support ``order`` option.

    Args:
        a (cupy.ndarray): Array to be flattened.

    Returns:
        cupy.ndarray: A flattened view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.ravel`

    """
    # TODO(beam2d): Support ordering option
    return reshape(a, a.size)

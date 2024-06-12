import numpy
if numpy.__version__ < '2':
    from numpy.lib import index_tricks
else:
    import numpy.lib._index_tricks_impl as index_tricks  # type: ignore[no-redef]  # NOQA

import cupy
from cupy._core import internal


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function to 1-D slices along the given axis.

    Args:
        func1d (function (M,) -> (Nj...)): This function should accept 1-D
            arrays. It is applied to 1-D slices of ``arr`` along the specified
            axis. It must return a 1-D ``cupy.ndarray``.
        axis (integer): Axis along which ``arr`` is sliced.
        arr (cupy.ndarray (Ni..., M, Nk...)): Input array.
        args: Additional arguments for ``func1d``.
        kwargs: Additional keyword arguments for ``func1d``.

    Returns:
        cupy.ndarray: The output array. The shape of ``out`` is identical to
        the shape of ``arr``, except along the ``axis`` dimension. This
        axis is removed, and replaced with new dimensions equal to the
        shape of the return value of ``func1d``. So if ``func1d`` returns a
        scalar ``out`` will have one fewer dimensions than ``arr``.

    .. seealso:: :func:`numpy.apply_along_axis`
    """
    ndim = arr.ndim
    axis = internal._normalize_axis_index(axis, ndim)
    inarr_view = cupy.moveaxis(arr, axis, -1)

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars
    inds = index_tricks.ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            'Cannot apply_along_axis when any iteration dimensions are 0'
        )
    res = func1d(inarr_view[ind0], *args, **kwargs)
    res = cupy.asarray(res)

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = cupy.empty(inarr_view.shape[:-1] + res.shape, res.dtype)

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        out = func1d(inarr_view[ind], *args, **kwargs)
        buff[ind] = cupy.asarray(out)

    # restore the inserted axes back to where they belong
    for i in range(res.ndim):
        buff = cupy.moveaxis(buff, -1, axis)

    return buff


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over

    if not cupy.issubdtype(indices.dtype, cupy.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions")

    shape_ones = (1, ) * indices.ndim
    dest_dims = list(range(axis)) + [None] + \
        list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal cupy.arange calls,
    # with the requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(cupy.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def put_along_axis(arr, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    Args:
        arr : cupy.ndarray (Ni..., M, Nk...)
            Destination array.
        indices : cupy.ndarray (Ni..., J, Nk...)
            Indices to change along each 1d slice of `arr`. This must match the
            dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
            against `arr`.
        values : array_like (Ni..., J, Nk...)
            values to insert at those indices. Its shape and dimension are
            broadcast to match that of `indices`.
        axis : int
            The axis to take 1d slices along. If axis is None, the destination
            array is treated as if a flattened 1d view had been created of it.

    .. seealso:: :func:`numpy.put_along_axis`
    """

    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise NotImplementedError(
                "Tuple setitem isn't supported for flatiter.")
        # put is roughly equivalent to a.flat[ind] = values
        cupy.put(arr, indices, values)
    else:
        axis = internal._normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

        # use the fancy index
        arr[_make_along_axis_idx(arr_shape, indices, axis)] = values

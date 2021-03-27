from numpy.lib import index_tricks

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

    .. seealso:: :func:`numpy.apply_over_axes`
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
    if cupy.isscalar(res):
        # scalar outputs need to be transfered to a device ndarray
        res = cupy.asarray(res)

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = cupy.empty(inarr_view.shape[:-1] + res.shape, res.dtype)

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = func1d(inarr_view[ind], *args, **kwargs)

    # restore the inserted axes back to where they belong
    for i in range(res.ndim):
        buff = cupy.moveaxis(buff, -1, axis)

    return buff

from numpy.lib import index_tricks

import cupy
from cupy._core import internal
from cupy._creation.from_data import array, asarray
from cupy._manipulation.dims import expand_dims
from cupy._manipulation.kind import asfarray


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

def apply_over_axes(func, a, axes):
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first
    element of `axes`.  The result `res` of the function call must have
    either the same dimensions as `a` or one less dimension.  If `res`
    has one less dimension than `a`, a dimension is inserted before
    `axis`.  The call to `func` is then repeated for each axis in `axes`,
    with `res` as the first argument.

    Parameters
    ----------
    func : function
        This function must take two arguments, `func(a, axis)`.
    a : array_like
        Input array.
    axes : array_like
        Axes over which `func` is applied; the elements must be integers.

    Returns
    -------
    apply_over_axis : ndarray
        The output array.  The number of dimensions is the same as `a`,
        but the shape can be different.  This depends on whether `func`
        changes the shape of its output with respect to its input.

    See Also
    --------
    apply_along_axis :
        Apply a function to 1-D slices of an array along the given axis.

    Notes
    -----
    This function is equivalent to tuple axis arguments to reorderable ufuncs
    with keepdims=True. Tuple axis arguments to ufuncs have been available.

    Examples
    --------
    >>> a = np.arange(24).reshape(2,3,4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])

    Sum over axes 0 and 2. The result has same number of dimensions
    as the original array:

    >>> np.apply_over_axes(np.sum, a, [0,2])
    array([[[ 60],
            [ 92],
            [124]]])

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.sum(a, axis=(0,2), keepdims=True)
    array([[[ 60],
            [ 92],
            [124]]])

    """
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        res = func(val, axis)
        if res.ndim == val.ndim:
            val = res
        else:
            res = expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                                 "an array of the correct shape")
    return val
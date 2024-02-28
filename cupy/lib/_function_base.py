import warnings

import cupy
from cupy._core import internal


def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : cupy.ndarray
        Input array.
    obj : int, slice or sequence of ints
        Object that defines the index or indices before which `values` is
        inserted.
    values : cupy.ndarray
        Values to insert into `arr`. If the type of `values` is different
        from that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that ``arr[...,obj,...] = values``
        is legal.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.

    Returns
    -------
    out : cupy.ndarray
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.

    Notes
    -----
    Note that for higher dimensional inserts `obj=0` behaves very different
    from `obj=[0]` just like `arr[:,0,:] = values` is different from
    `arr[:,[0],:] = values`.

    .. seealso:: :func:`numpy.insert`
    """

    arr = cupy.asarray(arr)
    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        # needed for np.matrix, which is still not 1d after being ravelled
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = internal._normalize_axis_index(axis, ndim)
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        indices = cupy.arange(*obj.indices(N), dtype=cupy.intp)
    else:
        # need to copy obj, because indices will be changed in-place
        indices = cupy.array(obj)
        if indices.dtype == bool:
            warnings.warn(
                'in the future insert will treat boolean arrays and '
                'array-likes as a boolean index instead of casting it to '
                'integer', FutureWarning, stacklevel=3)
            indices = indices.astype(cupy.intp)
        elif indices.ndim > 1:
            raise ValueError(
                'index array argument obj to insert must be one dimensional '
                'or scalar')

    if indices.size == 1:
        index = indices.item()
        if index < -N or index > N:
            raise IndexError(f'index {obj} is out of bounds for axis {axis} '
                             f'with size {N}')
        if (index < 0):
            index += N

        values = cupy.array(values, copy=False,
                            ndmin=arr.ndim, dtype=arr.dtype)
        if indices.ndim == 0:
            # broadcasting is very different here, since a[:,0,:] = ... behaves
            # very different from a[:,[0],:] = ...! This changes values so that
            # it works likes the second case. (here a[:,0:1,:])
            values = cupy.moveaxis(values, 0, axis)
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = cupy.empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index+numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index+numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]

        return new

    elif indices.size == 0 and not isinstance(obj, cupy.ndarray):
        indices = indices.astype(cupy.intp)

    indices[indices < 0] += N

    numnew = len(indices)
    order = indices.argsort()
    indices[order] += cupy.arange(numnew)

    newshape[axis] += numnew
    old_mask = cupy.ones(newshape[axis], dtype=bool)
    old_mask[indices] = False

    new = cupy.empty(newshape, arr.dtype, arrorder)
    slobj2 = [slice(None)]*ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = arr

    return new

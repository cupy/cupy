import cupy as cp

def median(a, axis=None, keepdims=False):
    r, k = _ureduce(a, func=_median, axis=axis)
    if keepdims:
        return r.reshape(k)
    else:
        return r

def normalize_axis_tuple(axis, nd):
    axs = cp.asarray(axis)
    a = (nd <= axs) * 1
    if 1 in a:
        print("Wrong!")
    else:
        norm = [int(ax) + 3 if ax < 0 else int(ax) for ax in axs]
    return tuple(norm)

def _ureduce(a, func, **kwargs):
    a = cp.asarray(a)
    axis = kwargs.get('axis', None)
    if axis is not None:
        keepdim = list(a.shape)
        nd = a.ndim
        axis = normalize_axis_tuple(axis, nd)

        for ax in axis:
            keepdim[ax] = 1

        if len(axis) == 1:
            kwargs['axis'] = axis[0]

        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
        keepdim = tuple(keepdim)
    else:
        keepdim = (1,) * a.ndim

    r = func(a, **kwargs)

    return r, keepdim

def _median(a, axis=None):
    a = cp.asarray(a)

    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    
    a.partition(kth, axis=axis)
    part = a
    
    part = cp.partition(a, kth, axis=axis)

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)

    return cp.mean(part[indexer], axis=axis)
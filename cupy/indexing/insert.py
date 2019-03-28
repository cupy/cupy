import numpy

import cupy


def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values.

    This function uses the first N elements of `vals`, where N is the number
    of true values in `mask`.

    Args:
        arr (cupy.ndarray): Array to put data into.
        mask (array-like): Boolean mask array. Must have the same size as `a`.
        vals (array-like): Values to put into `a`. Only the first
            N elements are used, where N is the number of True values in
            `mask`. If `vals` is smaller than N, it will be repeated, and if
            elements of `a` are to be masked, this sequence must be non-empty.

    Examples
    --------
    >>> arr = np.arange(6).reshape(2, 3)
    >>> np.place(arr, arr>2, [44, 55])
    >>> arr
    array([[ 0,  1,  2],
           [44, 55, 44]])

    .. seealso:: :func:`numpy.place`
    """
    mask = cupy.asarray(mask)
    if arr.size != mask.size:
        raise ValueError('Mask and data must be the same size.')
    vals = cupy.asarray(vals)

    mask_indices = mask.ravel().nonzero()[0]
    if mask_indices.size == 0:
        return
    if vals.size == 0:
        raise ValueError('Cannot insert from an empty array.')
    arr.put(mask_indices, vals, mode='wrap')


def put(a, ind, v, mode='wrap'):
    """Replaces specified elements of an array with given values.

    Args:
        a (cupy.ndarray): Target array.
        ind (array-like): Target indices, interpreted as integers.
        v (array-like): Values to place in `a` at target indices.
            If `v` is shorter than `ind` it will be repeated as necessary.
        mode (str): How out-of-bounds indices will behave. Its value must be
            either `'raise'`, `'wrap'` or `'clip'`. Otherwise,
            :class:`TypeError` is raised.

    .. note::
        Default `mode` is set to `'wrap'` to avoid unintended performance drop.
        If you need NumPy's behavior, please pass `mode='raise'` manually.

    .. seealso:: :func:`numpy.put`
    """
    a.put(ind, v, mode=mode)


# TODO(okuta): Implement putmask


def fill_diagonal(a, val, wrap=False):
    """Fills the main diagonal of the given array of any dimensionality.

    For an array `a` with ``a.ndim > 2``, the diagonal is the list of
    locations with indices ``a[i, i, ..., i]`` all identical. This function
    modifies the input array in-place, it does not return a value.

    Args:
        a (cupy.ndarray): The array, at least 2-D.
        val (scalar): The value to be written on the diagonal.
            Its type must be compatible with that of the array a.
        wrap (bool): If specified, the diagonal is "wrapped" after N columns.
            This affects only tall matrices.

    Examples
    --------
    >>> a = cupy.zeros((3, 3), int)
    >>> cupy.fill_diagonal(a, 5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

     .. seealso:: :func:`numpy.fill_diagonal`
    """
    # The followings are imported from the original numpy
    if a.ndim < 2:
        raise ValueError('array must be at least 2-d')
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not numpy.alltrue(numpy.diff(a.shape) == 0):
            raise ValueError('All dimensions of input must be of equal length')
        step = 1 + numpy.cumprod(a.shape[:-1]).sum()

    # Since the current cupy does not support a.flat,
    # we use a.ravel() instead of a.flat
    a.ravel()[:end:step] = val

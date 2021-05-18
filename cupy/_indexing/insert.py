import numpy

import cupy
from cupy import _core


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

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.place`
    """
    # TODO(niboshi): Avoid nonzero which may synchronize the device.
    mask = cupy.asarray(mask)
    if arr.size != mask.size:
        raise ValueError('Mask and data must be the same size.')
    vals = cupy.asarray(vals)

    mask_indices = mask.ravel().nonzero()[0]  # may synchronize
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


_putmask_kernel = _core.ElementwiseKernel(
    'Q mask, raw S values, uint64 len_vals', 'T out',
    '''
    if (mask) out = (T) values[i % len_vals];
    ''',
    'putmask_kernel'
)


def putmask(a, mask, values):
    """
    Changes elements of an array inplace, based on a conditional mask and
    input values.

    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.
    If `values` is not the same size as `a` and `mask` then it will repeat.

    Args:
        a (cupy.ndarray): Target array.
        mask (cupy.ndarray): Boolean mask array. It has to be
            the same shape as `a`.
        values (cupy.ndarray or scalar): Values to put into `a` where `mask`
            is True. If `values` is smaller than `a`, then it will be
            repeated.

    Examples
    --------
    >>> x = cupy.arange(6).reshape(2, 3)
    >>> cupy.putmask(x, x>2, x**2)
    >>> x
    array([[ 0,  1,  2],
           [ 9, 16, 25]])

    If `values` is smaller than `a` it is repeated:

    >>> x = cupy.arange(6)
    >>> cupy.putmask(x, x>2, cupy.array([-33, -44]))
    >>> x
    array([  0,   1,   2, -44, -33, -44])

    .. seealso:: :func:`numpy.putmask`

    """

    if not isinstance(a, cupy.ndarray):
        raise TypeError('`a` should be of type cupy.ndarray')
    if not isinstance(mask, cupy.ndarray):
        raise TypeError('`mask` should be of type cupy.ndarray')
    if not (cupy.isscalar(values) or isinstance(values, cupy.ndarray)):
        raise TypeError('`values` should be of type cupy.ndarray')

    if not a.shape == mask.shape:
        raise ValueError('mask and data must be the same size')

    mask = mask.astype(numpy.bool_)

    if cupy.isscalar(values):
        a[mask] = values

    elif not numpy.can_cast(values.dtype, a.dtype):
        raise TypeError('Cannot cast array data from'
                        ' {} to {} according to the rule \'safe\''
                        .format(values.dtype, a.dtype))

    elif a.shape == values.shape:
        a[mask] = values[mask]

    else:
        values = values.ravel()
        _putmask_kernel(mask, values, len(values), a)


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

    a.flat[:end:step] = val


def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array.

    Returns a tuple of indices that can be used to access the main
    diagonal of an array with ``ndim >= 2`` dimensions and shape
    (n, n, ..., n).

    Args:
        n (int): The size, along each dimension of the arrays for which
            the indices are to be returned.
        ndim (int): The number of dimensions. default `2`.

    Examples
    --------
    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> di = cupy.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = cupy.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Create indices to manipulate a 3-D array:

    >>> d3 = cupy.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = cupy.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [0, 1]]])

    .. seealso:: :func:`numpy.diag_indices`

    """
    idx = cupy.arange(n)
    return (idx,) * ndim


def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.
    See `diag_indices` for full details.

    Args:
        arr (cupy.ndarray): At least 2-D.

    .. seealso:: :func:`numpy.diag_indices_from`

    """
    if not isinstance(arr, cupy.ndarray):
        raise TypeError("Argument must be cupy.ndarray")

    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    if not cupy.all(cupy.diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(arr.shape[0], arr.ndim)

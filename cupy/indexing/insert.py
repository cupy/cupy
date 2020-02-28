import string

import numpy

import cupy
from cupy import util
from cupy.core import _carray
from cupy.core import _scalar
from cupy.cuda import device


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


# TODO(okuta): Implement putmask


_fill_diagonal_template = string.Template(r'''
#include <cupy/complex.cuh>
#include <cupy/carray.cuh>
extern "C" __global__
void cupy_fill_diagonal(CArray<${type}, ${a_ndim}> a,
                        CIndexer<${a_ndim}> a_ind,
                        int start,
                        int stop,
                        int step,
                        CArray<${type}, ${val_ndim}> val,
                        CIndexer<${val_ndim}> val_ind) {
    int n = (stop - start) / step + 1;
    CUPY_FOR(i, n) {
        a_ind.set(start + i * step);
        val_ind.set(i % val_ind.size());
        a[a_ind.get()] = val[val_ind.get()];
    }
}''')


@util.memoize(for_each_device=True)
def _fill_diagonal_kernel(type, a_ndim, val_ndim):
    code = _fill_diagonal_template.substitute(
        type=type, a_ndim=a_ndim, val_ndim=val_ndim)
    return cupy.RawKernel(code, 'cupy_fill_diagonal')


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
    end = a.size
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not numpy.alltrue(numpy.diff(a.shape) == 0):
            raise ValueError('All dimensions of input must be of equal length')
        step = 1 + numpy.cumprod(a.shape[:-1]).sum()

    val = cupy.asarray(val, dtype=a.dtype)

    dev_id = device.get_device_id()
    for arr in [a, val]:
        if arr.data.device_id != dev_id:
            raise ValueError(
                'Array device must be same as the current '
                'device: array device = %d while current = %d'
                % (arr.data.device_id, dev_id))

    typename = _scalar.get_typename(a.dtype)
    fill_diagonal_kernel = _fill_diagonal_kernel(typename, a.ndim, val.ndim)

    size = end // step + 1
    a_ind = _carray.Indexer(a.shape)
    val_ind = _carray.Indexer(val.shape)
    fill_diagonal_kernel.kernel.linear_launch(
        size, (a, a_ind, 0, end, step, val, val_ind))

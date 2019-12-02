import cupy
from cupy import core
from cupy.core import fusion

from cupy.core import _routines_statistics as _statistics


def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the maximum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis ``axis`` is preserved as an axis
            of length one.

    Returns:
        cupy.ndarray: The indices of the maximum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmax`

    """
    # TODO(okuta): check type
    return a.argmax(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanargmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slice ``-1`` is returned.
    Subclass cannot be passed yet, subok=True still unsupported

    Args:
        a (cupy.ndarray): Array to take nanargmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.

    Returns:
        cupy.ndarray: The indices of the maximum of ``a``
            along an axis ignoring NaN values.

    .. note:: For performance reasons, ``cupy.nanargmax`` returns
            ``out of range values`` for all-NaN slice
            whereas ``numpy.nanargmax`` raises ``ValueError``
    .. seealso:: :func:`numpy.nanargmax`
    """
    if a.dtype.kind in 'biu':
        return argmax(a, axis=axis)

    return _statistics._nanargmax(a, axis, dtype, out, keepdims)


def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the minimum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis ``axis`` is preserved as an axis
            of length one.

    Returns:
        cupy.ndarray: The indices of the minimum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmin`

    """
    # TODO(okuta): check type
    return a.argmin(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanargmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slice ``-1`` is returned.
    Subclass cannot be passed yet, subok=True still unsupported

    Args:
        a (cupy.ndarray): Array to take nanargmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.

    Returns:
        cupy.ndarray: The indices of the minimum of ``a``
            along an axis ignoring NaN values.

    .. note:: For performance reasons, ``cupy.nanargmin`` returns
            ``out of range values`` for all-NaN slice
            whereas ``numpy.nanargmin`` raises ``ValueError``
    .. seealso:: :func:`numpy.nanargmin`
    """
    if a.dtype.kind in 'biu':
        return argmin(a, axis=axis)

    return _statistics._nanargmin(a, axis, dtype, out, keepdims)

# TODO(okuta): Implement argwhere


def nonzero(a):
    """Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of a,
    containing the indices of the non-zero elements in that dimension.

    Args:
        a (cupy.ndarray): array

    Returns:
        tuple of arrays: Indices of elements that are non-zero.

    .. seealso:: :func:`numpy.nonzero`

    """
    assert isinstance(a, core.ndarray)
    return a.nonzero()


def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a.

    This is equivalent to a.ravel().nonzero()[0].

    Args:
        a (cupy.ndarray): input array

    Returns:
        cupy.ndarray: Output array,
        containing the indices of the elements of a.ravel() that are non-zero.

    .. seealso:: :func:`numpy.flatnonzero`
    """
    assert isinstance(a, core.ndarray)
    return a.ravel().nonzero()[0]


_where_ufunc = core.create_ufunc(
    'cupy_where',
    ('???->?', '?bb->b', '?BB->B', '?hh->h', '?HH->H', '?ii->i', '?II->I',
     '?ll->l', '?LL->L', '?qq->q', '?QQ->Q', '?ee->e', '?ff->f',
     # On CUDA 6.5 these combinations don't work correctly (on CUDA >=7.0, it
     # works).
     # See issue #551.
     '?hd->d', '?Hd->d',
     '?dd->d', '?FF->F', '?DD->D'),
    'out0 = in0 ? in1 : in2')


def where(condition, x=None, y=None):
    """Return elements, either from x or y, depending on condition.

    If only condition is given, return ``condition.nonzero()``.

    Args:
        condition (cupy.ndarray): When True, take x, otherwise take y.
        x (cupy.ndarray): Values from which to choose on ``True``.
        y (cupy.ndarray): Values from which to choose on ``False``.

    Returns:
        cupy.ndarray: Each element of output contains elements of ``x`` when
            ``condition`` is ``True``, otherwise elements of ``y``. If only
            ``condition`` is given, return the tuple ``condition.nonzero()``,
            the indices where ``condition`` is True.

    .. seealso:: :func:`numpy.where`

    """

    missing = (x is None, y is None).count(True)

    if missing == 1:
        raise ValueError('Must provide both \'x\' and \'y\' or neither.')
    if missing == 2:
        return nonzero(condition)

    if fusion._is_fusing():
        return fusion._call_ufunc(_where_ufunc, condition, x, y)
    return _where_ufunc(condition.astype('?'), x, y)


# This is to allow using the same kernels for all dtypes, ints & floats
# as nan is a special case
_preamble = '''
template<typename T>
__device__ bool _isnan(T val) {
    return false;
}
template<>
__device__ bool _isnan<float16>(float16 val) {
    return isnan(val);
}
template<>
__device__ bool _isnan<float>(float val) {
    return isnan(val);
}
template<>
__device__ bool _isnan<double>(double val) {
    return isnan(val);
}
template<>
__device__ bool _isnan<const complex<double>&>(const complex<double>& val) {
    return isnan(val);
}
template<>
__device__ bool _isnan<const complex<float>&>(const complex<float>& val) {
    return isnan(val);
}
'''

_searchsorted_kernel_left = core.ElementwiseKernel(
    'S x, raw T bins, int32 n_bins',
    'U y',
    '''
    y = 0;
    // Array is assumed to be monotonically
    // increasing unless a check is requested
    // because of functions like digitize
    // allowing both, increasing and decreasing.
    bool inc = true;
    if (_isnan<S>(x)) {
        y = (inc ? n_bins : 0);
        return;
    }
    if (inc && x > bins[n_bins-1] || !inc && x <= bins[n_bins-1]) {
        y = n_bins;
        return;
    }
    size_t l = 0;
    size_t r = n_bins-1;
    while (l<r) {
        size_t m = l + (r - l) / 2;
        if ((inc && bins[m] >= x) || (!inc && bins[m] < x)) {
            r = m;
        } else {
            l = m + 1;
        }
    }
    y = r;
    ''', preamble=_preamble)


_searchsorted_kernel_right = core.ElementwiseKernel(
    'S x, raw T bins, int32 n_bins',
    'U y',
    '''
    y = 0;
    // Array is assumed to be monotonically
    // increasing unless a check is requested
    // because of functions like digitize
    // allowing both, increasing and decreasing.
    bool inc = true;
    if(_isnan<S>(x)) {
        y = (inc ? n_bins : 0);
        return;
    }
    if (inc && x >= bins[n_bins-1]) {
        y = n_bins;
        return;
    }
    size_t l = 0;
    size_t r = n_bins-1;
    while (l<r) {
        size_t m = l + (r - l) / 2;
        if ((inc && bins[m] <= x) || (!inc && bins[m] > x)) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    y = r;
    ''', preamble=_preamble)


def searchsorted(a, v, side='left', sorter=None):
    """Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array a such that,
    if the corresponding elements in v were inserted before the indices,
    the order of a would be preserved.

    Args:
        a (cupy.ndarray): Input array. If ``sorter`` is None, then
            it must be sorted in ascending order,
            otherwise ``sorter`` must be an array of indices that sort it.
        v (array like): Values to insert into a.
        side : {'left', 'right'}
            If ``left``, return the index of the first suitable location found
            If ``right``, return the last such index.
            If there is no suitable index, return either 0 or length of ``a``.
        sorter : 1-D array_like
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.


    Returns:
        cupy.ndarray: Array of insertion points with the same shape as `v`.

    .. note:: When a is not in ascending order, behavior is undefined.
       NumPy avoids this check for efficiency

    .. seealso:: :func:`numpy.searchsorted`

    """
    return searchsorted_internal(a, v, side, sorter)


def searchsorted_internal(a, v, side, sorter):

    if not isinstance(a, cupy.ndarray):
        raise NotImplementedError('Only int or ndarray are supported for a')

    if not isinstance(v, cupy.ndarray):
        raise NotImplementedError('Only int or ndarray are supported for v')

    if a.ndim > 1:
        raise ValueError('object too deep for desired array')
    if a.ndim < 1:
        raise ValueError('object of too small depth for desired array')
    if a.size == 0:
        return cupy.zeros(v.shape, dtype=cupy.int64)

    a_iscomplex = a.dtype.kind == 'c'
    v_iscomplex = v.dtype.kind == 'c'

    if a_iscomplex and not v_iscomplex:
        v = v.astype(a.dtype)
    elif v_iscomplex and not a_iscomplex:
        a = a.astype(v.dtype)

    # Numpy does not check if the array is monotonic inside searchsorted
    # which leds to undefined behavior in such cases.
    if sorter is not None:
        if sorter.dtype.kind not in ('i', 'u'):
            raise TypeError('sorter must be of integer type')
        if sorter.size != a.size:
            raise ValueError('sorter.size must equal a.size')
        a = a.take(sorter)

    # NumPy digitize reverses the array when its monotonically decreasing.
    # For CUDA it's better to change the comparisons in the kernel rather
    # than reversingly reading the array or use take.
    y = cupy.zeros(v.shape, dtype=cupy.int64)
    if side == 'right':
        _searchsorted_kernel_right(v, a, a.size, y)
    else:
        _searchsorted_kernel_left(v, a, a.size, y)
    return y


# TODO(okuta): Implement extract

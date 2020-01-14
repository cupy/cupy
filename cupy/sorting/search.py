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

    .. note::
       ``dtype`` and ``keepdim`` arguments are specific to CuPy. They are
       not in NumPy.

    .. note::
       ``axis`` argument accepts a tuple of ints, but this is specific to
       CuPy. NumPy does not support it.

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

    .. note::
       ``dtype`` and ``keepdim`` arguments are specific to CuPy. They are
       not in NumPy.

    .. note::
       ``axis`` argument accepts a tuple of ints, but this is specific to
       CuPy. NumPy does not support it.

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

    .. warning::

        This function may synchronize the device.

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

    .. warning::

        This function may synchronize the device.

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

    .. warning::

        This function may synchronize the device if both ``x`` and ``y`` are
        omitted.

    .. seealso:: :func:`numpy.where`

    """

    missing = (x is None, y is None).count(True)

    if missing == 1:
        raise ValueError('Must provide both \'x\' and \'y\' or neither.')
    if missing == 2:
        return nonzero(condition)  # may synchronize

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
__device__ bool _isnan(float16 val) {
    return isnan(val);
}
template<>
__device__ bool _isnan(float val) {
    return isnan(val);
}
template<>
__device__ bool _isnan(double val) {
    return isnan(val);
}
template<>
__device__ bool _isnan(const complex<double>& val) {
    return isnan(val);
}
template<>
__device__ bool _isnan(const complex<float>& val) {
    return isnan(val);
}
'''


_searchsorted_kernel = core.ElementwiseKernel(
    'S x, raw T bins, int64 n_bins, bool side_is_right',
    'int64 y',
    '''
    if (_isnan<S>(x)) {
        long long pos = n_bins;
        if (!side_is_right) {
            while (pos > 0 && _isnan<T>(bins[pos-1])) {
                --pos;
            }
        }
        y = pos;
        return;
    }
    bool greater = (side_is_right ? x >= bins[n_bins-1] : x > bins[n_bins-1]);
    if (greater) {
        y = n_bins;
        return;
    }
    long long left = 0;
    long long right = n_bins-1;
    while (left < right) {
        long long m = left + (right - left) / 2;
        if (side_is_right ? bins[m] <= x : bins[m] < x) {
            left = m + 1;
        } else {
            right = m;
        }
    }
    y = right;
    ''', preamble=_preamble)


def searchsorted(a, v, side='left', sorter=None):
    """Finds indices where elements should be inserted to maintain order.

    Find the indices into a sorted array ``a`` such that,
    if the corresponding elements in ``v`` were inserted before the indices,
    the order of ``a`` would be preserved.

    Args:
        a (cupy.ndarray): Input array. If ``sorter`` is ``None``, then
            it must be sorted in ascending order,
            otherwise ``sorter`` must be an array of indices that sort it.
        v (cupy.ndarray): Values to insert into ``a``.
        side : {'left', 'right'}
            If ``left``, return the index of the first suitable location found
            If ``right``, return the last such index.
            If there is no suitable index, return either 0 or length of ``a``.
        sorter : 1-D array_like
            Optional array of integer indices that sort array ``a`` into
            ascending order. They are typically the result of
            :func:`~cupy.argsort`.

    Returns:
        cupy.ndarray: Array of insertion points with the same shape as ``v``.

    .. note:: When a is not in ascending order, behavior is undefined.

    .. seealso:: :func:`numpy.searchsorted`

    """

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

    y = cupy.zeros(v.shape, dtype=cupy.int64)
    _searchsorted_kernel(v, a, a.size, side == 'right', y)
    return y


# TODO(okuta): Implement extract

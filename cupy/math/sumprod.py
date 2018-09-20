import numpy
import six

import cupy
from cupy import core
from cupy.core import fusion


@fusion._reduction_wrapper(core.core._sum_auto_dtype)
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.sum`

    """
    # TODO(okuta): check type
    return a.sum(axis, dtype, out, keepdims)


@fusion._reduction_wrapper(core.core._prod_auto_dtype)
def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.prod`

    """
    # TODO(okuta): check type
    return a.prod(axis, dtype, out, keepdims)


# TODO(okuta): Implement nansum


def _axis_to_first(x, axis):
    if axis < 0:
        axis = x.ndim + axis
    trans = [axis] + [a for a in six.moves.range(x.ndim) if a != axis]
    pre = list(six.moves.range(1, axis + 1))
    succ = list(six.moves.range(axis + 1, x.ndim))
    revert = pre + [0] + succ
    return trans, revert


def _proc_as_batch(proc, x, axis):
    if x.shape[axis] == 0:
        return cupy.empty_like(x)
    trans, revert = _axis_to_first(x, axis)
    t = x.transpose(trans)
    s = t.shape
    r = t.reshape(x.shape[axis], -1)
    pos = 1
    size = r.size
    batch = r.shape[1]
    while pos < size:
        proc(pos, batch, r, size=size)
        pos <<= 1
    return r.reshape(s).transpose(revert)


def _cum_core(a, axis, dtype, out, kern, batch_kern):
    if out is None:
        if dtype is None:
            kind = a.dtype.kind
            if kind == 'b':
                dtype = numpy.dtype('l')
            elif kind == 'i' and a.dtype.itemsize < numpy.dtype('l').itemsize:
                dtype = numpy.dtype('l')
            elif kind == 'u' and a.dtype.itemsize < numpy.dtype('L').itemsize:
                dtype = numpy.dtype('L')
            else:
                dtype = a.dtype

        out = a.astype(dtype)
    else:
        out[...] = a

    if axis is None:
        out = out.ravel()
    elif not (-a.ndim <= axis < a.ndim):
        raise core.core._AxisError('axis(={}) out of bounds'.format(axis))
    else:
        return _proc_as_batch(batch_kern, out, axis=axis)

    pos = 1
    while pos < out.size:
        kern(pos, out, size=out.size)
        pos <<= 1
    return out


_cumsum_batch_kern = core.ElementwiseKernel(
    'int64 pos, int64 batch', 'raw T x',
    '''
    ptrdiff_t b = i % batch;
    ptrdiff_t j = i / batch;
    if (j & pos) {
      const ptrdiff_t dst_index[] = {j, b};
      const ptrdiff_t src_index[] = {j ^ pos | (pos - 1), b};
      x[dst_index] += x[src_index];
    }
    ''',
    'cumsum_batch_kernel'
)
_cumsum_kern = core.ElementwiseKernel(
    'int64 pos', 'raw T x',
    '''
    if (i & pos) {
      x[i] += x[i ^ pos | (pos - 1)];
    }
    ''',
    'cumsum_kernel'
)


def cumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative sum is taken. If it is not
            specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumsum`

    """
    return _cum_core(a, axis, dtype, out, _cumsum_kern, _cumsum_batch_kern)


_cumprod_batch_kern = core.ElementwiseKernel(
    'int64 pos, int64 batch', 'raw T x',
    '''
    ptrdiff_t b = i % batch;
    ptrdiff_t j = i / batch;
    if (j & pos) {
      const ptrdiff_t dst_index[] = {j, b};
      const ptrdiff_t src_index[] = {j ^ pos | (pos - 1), b};
      x[dst_index] *= x[src_index];
    }
    ''',
    'cumprod_batch_kernel'
)
_cumprod_kern = core.ElementwiseKernel(
    'int64 pos', 'raw T x',
    '''
    if (i & pos) {
      x[i] *= x[i ^ pos | (pos - 1)];
    }
    ''',
    'cumprod_kernel'
)


def cumprod(a, axis=None, dtype=None, out=None):
    """Returns the cumulative product of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative product is taken. If it is
            not specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumprod`

    """
    return _cum_core(a, axis, dtype, out, _cumprod_kern, _cumprod_batch_kern)


# TODO(okuta): Implement diff


# TODO(okuta): Implement ediff1d


# TODO(okuta): Implement gradient


# TODO(okuta): Implement cross


# TODO(okuta): Implement trapz

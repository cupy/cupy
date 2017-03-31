import numpy
import six

from cupy import core


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


# TODO(okuta): Implement cumprod


def _axis_to_first(x, axis):
    if axis < 0:
        axis = x.ndim + axis
    trans = [axis] + [a for a in six.moves.range(x.ndim) if a != axis]
    pre = list(six.moves.range(1, axis + 1))
    succ = list(six.moves.range(axis + 1, x.ndim))
    revert = pre + [0] + succ
    return trans, revert


def _proc_as_batch(proc, x, axis):
    trans, revert = _axis_to_first(x, axis)
    t = x.transpose(trans)
    s = t.shape
    r = t.reshape(x.shape[axis], -1).T
    result = proc(r)
    return result.T.reshape(s).transpose(revert)


def _cumsum_batch(out):
    kern = core.ElementwiseKernel(
        'int32 pos, int32 batch', 'raw T x',
        '''
        int b = i % batch;
        int j = i / batch;
        if (j & pos) {
          const int dst_index[] = {b, j};
          const int src_index[] = {b, j ^ pos | (pos - 1)};
          x[dst_index] += x[src_index];
        }
        ''',
        'cumsum_batch_kernel'
    )

    pos = 1
    while pos < out.size:
        kern(pos, out.shape[0], out, size=out.size)
        pos <<= 1
    return out


def cumsum(a, axis=None, dtype=None, out=None):
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
        raise ValueError('axis(={}) out of bounds'.format(axis))
    else:
        return _proc_as_batch(_cumsum_batch, out, axis=axis)

    kern = core.ElementwiseKernel(
        'int32 pos', 'raw T x',
        '''
        if (i & pos) {
          x[i] += x[i ^ pos | (pos - 1)];
        }
        ''',
        'cumsum_kernel'
    )

    pos = 1
    while pos < out.size:
        kern(pos, out, size=out.size)
        pos <<= 1
    return out


# TODO(okuta): Implement diff


# TODO(okuta): Implement ediff1d


# TODO(okuta): Implement gradient


# TODO(okuta): Implement cross


# TODO(okuta): Implement trapz

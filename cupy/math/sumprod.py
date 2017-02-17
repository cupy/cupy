import numpy

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


def cumsum(a, axis=None, dtype=None, out=None):
    if axis is None:
        a = a.ravel()
    else:
        raise ValueError("'axis' option is not supported")

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

def take(a, indices, axis=None, out=None):
    """Takes elements of an array at specified indices along an axis.

    This is an implementation of "fancy indexing" at single axis.

    This function does not support ``mode`` option.

    Args:
        a (cupy.ndarray): Array to extract elements.
        indices (int or array-like): Indices of elements that this function
            takes.
        axis (int): The axis along which to select indices. The flattened input
            is used by default.
        out (cupy.ndarray): Output array. If provided, it should be of
            appropriate shape and dtype.

    Returns:
        cupy.ndarray: The result of fancy indexing.

    .. seealso:: :func:`numpy.take`

    """
    # TODO(okuta): check type
    return a.take(indices, axis, out)


def choose(a, choices, out=None, mode='raise'):
    n = choices.shape[0]

    if a.ndim + 1 < choices.ndim:
        for i in range(choices.ndim - (a.ndim + 1)):
            a = cupy.expand_dims(a, 0)
    elif a.ndim + 1 > choices.ndim:
        for i in range(a.ndim + 1 - choices.ndim):
            choices = cupy.expand_dims(choices, 1)
    ba, bcs = cupy.broadcast_arrays(a, choices)

    n_channel = numpy.prod(bcs[0].shape)
    if mode == 'raise':
        if not ((a < n).all() and (0 <= a).all()):
            raise ValueError('invalid entry in choice array')
        c = _choose_kernel(ba[0], bcs, n_channel)
    elif mode == 'wrap':
        ba = ba[0] % n
        c = _choose_kernel(ba, bcs, n_channel)
    elif mode == 'clip':
        c = _choose_clip_kernel(ba[0], bcs, n_channel, n)
    else:
        raise TypeError('clipmode not understood')

    return c


# TODO(okuta): Implement compress


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Returns specified diagonals.

    This function extracts the diagonals along two specified axes. The other
    axes are not changed. This function returns a writable view of this array
    as NumPy 1.10 will do.

    Args:
        a (cupy.ndarray): Array from which the diagonals are taken.
        offset (int): Index of the diagonals. Zero indicates the main
            diagonals, a positive value upper diagonals, and a negative value
            lower diagonals.
        axis1 (int): The first axis to take diagonals from.
        axis2 (int): The second axis to take diagonals from.

    Returns:
        cupy.ndarray: A view of the diagonals of ``a``.

    .. seealso:: :func:`numpy.diagonal`

    """
    # TODO(okuta): check type
    return a.diagonal(offset, axis1, axis2)


# TODO(okuta): Implement select


_take_kernel = elementwise.ElementwiseKernel(
    'raw T a, S indices, int64 cdim, int64 rdim',
    'T out',
    '''
      long long li = i / (rdim * cdim);
      long long ri = i % rdim;
      out = a[(li * cdim + indices) * rdim + ri];
    ''',
    'cupy_take')

_choose_kernel = elementwise.ElementwiseKernel(
    'S a, raw T choices, int32 n_channel',
    'T y',
    'y = choices[i + n_channel * a]',
    'cupy_choose')

_choose_clip_kernel = elementwise.ElementwiseKernel(
    'S a, raw T choices, int32 n_channel, int32 n',
    'T y',
    '''
      S x = a;
      if (a < 0) {
        x = 0;
      } else if (a >= n) {
        x = n - 1;
      }
      y = choices[i + n_channel * x];
    ''',
    'cupy_choose_clip')

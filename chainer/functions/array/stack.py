from chainer.functions.array import concat
from chainer.functions.array import expand_dims


def stack(xs, axis=0):
    """Concatenate variables along a new axis.

    Args:
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variables to be concatenated. The variables must have the
            same shape.
        axis (int): The axis along which the arrays will be stacked. The
            ``axis`` parameter is acceptable when
            :math:`-ndim - 1 \\leq axis \\leq ndim`. (``ndim`` is the
            dimension of input variables). When :math:`axis < 0`, the result
            is the same with :math:`ndim + 1 - |axis|`.

    Returns:
        ~chainer.Variable:
            Output variable. Let ``x_1, x_2, ..., x_n`` and ``y`` be the input
            variables and the output variable,
            ``y[:, ..., 0, ..., :]`` is ``x_1``,
            ``y[:, ..., 1, ..., :]`` is ``x_2``
            and ``y[:, ..., n-1, ..., :]`` is ``x_n`` (The indexed axis
            indicates the ``axis``).

    .. admonition:: Example

        >>> x1 = np.arange(0, 12).reshape(3, 4)
        >>> x1.shape
        (3, 4)
        >>> x1
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> x2 = np.arange(12, 24).reshape(3, 4)
        >>> x2.shape
        (3, 4)
        >>> x2
        array([[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
        >>> y = F.stack([x1, x2], axis=0)
        >>> y.shape
        (2, 3, 4)
        >>> y.data
        array([[[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]],
        <BLANKLINE>
               [[12, 13, 14, 15],
                [16, 17, 18, 19],
                [20, 21, 22, 23]]])
        >>> y = F.stack([x1, x2], axis=1)
        >>> y.shape
        (3, 2, 4)
        >>> y.data
        array([[[ 0,  1,  2,  3],
                [12, 13, 14, 15]],
        <BLANKLINE>
               [[ 4,  5,  6,  7],
                [16, 17, 18, 19]],
        <BLANKLINE>
               [[ 8,  9, 10, 11],
                [20, 21, 22, 23]]])
        >>> y = F.stack([x1, x2], axis=2)
        >>> y.shape
        (3, 4, 2)
        >>> y.data
        array([[[ 0, 12],
                [ 1, 13],
                [ 2, 14],
                [ 3, 15]],
        <BLANKLINE>
               [[ 4, 16],
                [ 5, 17],
                [ 6, 18],
                [ 7, 19]],
        <BLANKLINE>
               [[ 8, 20],
                [ 9, 21],
                [10, 22],
                [11, 23]]])
        >>> y = F.stack([x1, x2], axis=-1)
        >>> y.shape
        (3, 4, 2)

    """
    xs = [expand_dims.expand_dims(x, axis=axis) for x in xs]
    return concat.concat(xs, axis=axis)

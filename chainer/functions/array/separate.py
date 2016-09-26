from chainer.functions.array import reshape
from chainer.functions.array import split_axis


def separate(x, axis=0):
    """Separates an array along a given axis.

    This function separates an array along a given axis. For example, shape of
    an array is ``(2, 3, 4)``. When it separates the array with ``axis=1``, it
    returns three ``(2, 4)`` arrays.

    This function is an inverse of :func:`chainer.functions.stack`.

    Args:
        x (chainer.Variable): Variable to be separated.
        axis (int): Axis along which variables are separated.

    Returns:
        tuple of chainer.Variable: Output variables.

    .. seealso:: :func:`chainer.functions.stack`

    """
    shape = list(x.shape)
    del shape[axis]
    ys = split_axis.split_axis(x, x.shape[axis], axis, force_tuple=True)
    return tuple(reshape.reshape(y, shape) for y in ys)

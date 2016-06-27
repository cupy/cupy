from chainer.functions.array import concat
from chainer.functions.array import expand_dims


def stack(xs, axis=0):
    """Concatenate variables along a new axis.

    Args:
        xs (list of chainer.Variable): Variables to be concatenated.
        axis (int): Axis of result along which variables are stacked.

    Returns:
        ~chainer.Variable: Output variable.

    """
    xs = [expand_dims.expand_dims(x, axis=axis) for x in xs]
    return concat.concat(xs, axis=axis)

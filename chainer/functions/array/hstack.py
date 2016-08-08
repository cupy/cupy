from chainer.functions.array import concat
from chainer.functions.array import expand_dims


def hstack(xs):
    """Concatenate variables horizontally (column wise).

    Args:
        xs (list of chainer.Variable): Variables to be concatenated.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if xs[0].data.ndim == 0:
        xs = [expand_dims.expand_dims(x, 0) for x in xs]
    if xs[0].data.ndim == 1:
        return concat.concat(xs, axis=0)
    else:
        return concat.concat(xs, axis=1)

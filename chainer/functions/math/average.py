from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.math import sum as sum_mod


def average(x, axis=None, weights=None, keepdims=False):
    """Calculate weighted average of array elements over a given axis.

    Args:
        x (~chainer.Variable): Elements to sum.
        axis (None or int): Axis which the method is performed.
            With the default (axis = None) it performs a mean over all the
            dimensions of the input array.
        weights (None or chainer.Variable): An array holding weights to
            calculate weighted average. If it is ``None``, all weights are
            assumed to be one.
            When ``axis`` is ``None``, ``weights`` must have the same shape
            of ``x``. And when ``axis`` is ``int``, it must be 1-D array
            satisfing ``weights.shape == (x.shape[axis],)``.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if weights is not None:
        divider = sum_mod.sum(weights)
        if axis is not None:
            if axis < 0:
                axis += x.ndim
            w_shape = [d if i == axis else 1 for i, d in enumerate(x.shape)]
            weights = broadcast.broadcast_to(
                reshape.reshape(weights, w_shape), x.shape)

        x = x * weights
    else:
        if axis is None:
            divider = x.size
        else:
            divider = x.shape[axis]

    x_sum = sum_mod.sum(x, axis, keepdims)
    if weights is not None:
        # We do not need to call broadcast whene weights is None because
        # divider here is not a Variable but a scalar
        divider = broadcast.broadcast_to(divider, x_sum.shape)
    return x_sum / divider

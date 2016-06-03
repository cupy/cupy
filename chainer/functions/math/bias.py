from chainer.functions.array import reshape, broadcast


def bias(x, y, axis=1):
    """Elementwise summation with broadcasting.

    Computes a elementwise summation of two input variables, with the shape of
    the latter variable broadcasted to match the shape of the former. `axis`
    is the first axis of the first variable along which to apply the second
    variable.

    Args:
        x (~chainer.Variable): Input variable to be scaled.
        y (~chainer.Variable): Input variable to scale, broadcasted.
        axis (int): The first axis of `x` along which to apply `y`

    Returns:
        ~chainer.Variable: Output variable.

    """
    x_shape = x.data.shape
    y_shape = y.data.shape
    assert x_shape[axis:axis + len(y_shape)] == y_shape
    y1_shape = tuple([1] * axis + list(y_shape) +
                     [1] * (len(x_shape) - axis - len(y_shape)))
    y1 = reshape.reshape(y, y1_shape)
    y2 = broadcast.broadcast_to(y1, x_shape)
    return x + y2

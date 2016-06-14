from chainer.functions.array import broadcast
from chainer.functions.array import reshape


def scale(x, y, axis=1):
    """Elementwise product with broadcasting.

    Computes a elementwise product of two input variables, with the shape of
    the latter variable broadcasted to match the shape of the former. `axis`
    is the first axis of the first variable along which the second variable is
    applied.

    Args:
        x (~chainer.Variable): Input variable to be scaled.
        y (~chainer.Variable): Input variable to scale, broadcasted.
        axis (int): The first axis of `x` along which `y` is applied.

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
    return x * y2

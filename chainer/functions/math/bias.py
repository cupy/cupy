import chainer
from chainer.functions.array import broadcast
from chainer.functions.array import reshape


def bias(x, y, axis=1):
    """Elementwise summation with broadcasting.

    Computes a elementwise summation of two input variables, with the shape of
    the latter variable broadcasted to match the shape of the former. ``axis``
    is the first axis of the first variable along which the second variable is
    applied.

    The term "broadcasting" here comes from Caffe's bias layer so the
    "broadcasting" with the following arguments::

           x : 100 x 3 x 40 x 60
           y : 3 x 40
        axis : 1

    is equivalent to the following numpy broadcasting::

        x : 100 x 3 x 40 x 60
        y :   1 x 3 x 40 x 1

    Note that how the ``axis`` indicates to which axis of ``x`` we apply ``y``.

    Args:
        x (~chainer.Variable): Input variable to be summed.
        y (~chainer.Variable): Input variable to sum, broadcasted.
        axis (int): The first axis of ``x`` along which ``y`` is applied.

    Returns:
        ~chainer.Variable: Output variable.

    """
    x_shape = x.shape
    y_shape = y.shape
    if chainer.is_debug():
        assert x_shape[axis:axis + len(y_shape)] == y_shape
    y1_shape = tuple([1] * axis + list(y_shape) +
                     [1] * (len(x_shape) - axis - len(y_shape)))
    y1 = reshape.reshape(y, y1_shape)
    y2 = broadcast.broadcast_to(y1, x_shape)
    return x + y2

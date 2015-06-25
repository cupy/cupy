from chainer import Function


class Reshape(Function):

    """Reshapes an input array without copy."""

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x[0].reshape(self.shape),

    def backward(self, x, gy):
        return gy[0].reshape(x[0].shape),


def reshape(x, shape):
    """Reshapes an input variable without copy.

    Args:
        x (~chainer.Variable): Input variable.
        shape (tuple of ints): Target shape.

    Returns:
        ~chainer.Variable: Variable that holds a reshaped version of the input
            variable.

    """
    return Reshape(shape)(x)

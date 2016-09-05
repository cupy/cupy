from chainer import function


class Flatten(function.Function):

    """Flatten function."""

    def forward(self, inputs):
        return inputs[0].ravel(),

    def backward(self, inputs, grads):
        return grads[0].reshape(inputs[0].shape),


def flatten(x):
    """Flatten a given array.

    Args:
        x (~chainer.Varaiable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Flatten()(x)

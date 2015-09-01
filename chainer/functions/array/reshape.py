import numpy

from chainer import function
from chainer.utils import type_check


_type_check_prod = type_check.Variable(numpy.prod, 'prod')


class Reshape(function.Function):

    """Reshapes an input array without copy."""

    def __init__(self, shape):
        self.shape = shape

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            _type_check_prod(in_types[0].shape) ==
            _type_check_prod(self.shape)
        )

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

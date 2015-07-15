import numpy

from chainer import function
from chainer.utils import type_check


class Reshape(function.Function):

    """Reshapes an input array without copy."""

    def __init__(self, shape):
        self.shape = shape

    def check_type_forward(self, in_type):
        type_check.expect(in_type.size() == 1)
        x_type, = in_type

        in_shape_size = type_check.Variable(
            numpy.prod(x_type.shape.eval()), 'in_shape_size')
        out_shape_size = type_check.Variable(
            numpy.prod(self.shape), 'out_shape_size')
        type_check.expect(in_shape_size == out_shape_size)

    def check_type_backward(self, in_types, out_types):
        type_check.expect(out_types.size() == 1)
        x_type, = in_types
        y_type, = out_types

        in_shape_size = type_check.Variable(
            numpy.prod(x_type.shape.eval()), 'in_shape_size')
        out_shape_size = type_check.Variable(
            numpy.prod(y_type.shape.eval()), 'out_shape_size')
        type_check.expect(in_shape_size == out_shape_size)

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

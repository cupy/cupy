from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ExpandDims(function.Function):

    """Expands dimenstions of an input array without copy."""

    def __init__(self, axis):
        self.axis = int(axis)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        if self.axis >= 0:
            type_check.expect(x_type.ndim >= self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis - 1)

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return xp.expand_dims(x[0], self.axis),

    def backward(self, x, gy):
        return gy[0].reshape(x[0].shape),


def expand_dims(x, axis):
    """Expands dimensions of an input variable without copy.

    Args:
        x (~chainer.Variable): Input variable.
        axis (int): Position where new axis is to be inserted.

    Returns:
        ~chainer.Variable: Variable that holds a expanded input.
    """
    return ExpandDims(axis)(x)

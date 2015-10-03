from chainer import function
from chainer.utils import type_check


class Swapaxes(function.Function):
    """Swap two axes of an array."""

    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @property
    def label(self):
        return 'Swapaxes'

    def forward(self, inputs):
        x = inputs[0]
        y = x.swapaxes(self.axis1, self.axis2)
        return y,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        gx = gy.swapaxes(self.axis1, self.axis2)
        return gx,


def swapaxes(x, axis1, axis2):
    """Swap two axes of a variable.

    Args:
        x (~chainer.Variable): Input variable.
        axis1 (int): The first axis to swap.
        axis2 (int): The second axis to swap.

    Returns:
        ~chainer.Variable: Variable whose axes are swapped.

    """
    return Swapaxes(axis1, axis2)(x)

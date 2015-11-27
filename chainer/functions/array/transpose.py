import numpy

from chainer import function
from chainer.utils import type_check


class Transpose(function.Function):
    """Permute the dimensions of an array."""

    def __init__(self, axes=None):
        self.axes = axes

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @property
    def label(self):
        return 'Transpose'

    def forward(self, inputs):
        x = inputs[0]
        y = x.transpose(self.axes)
        return y,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        inv_axes = self.axes
        if self.axes:
            axes = tuple(ax % len(self.axes) for ax in self.axes)
            inv_axes = tuple(numpy.argsort(axes))
        gx = gy.transpose(inv_axes)
        return gx,


def transpose(x, axes=None):
    """Permute the dimensions of an input variable without copy.

    Args:
        x (~chainer.Variable): Input variable.
        axes (tuple of ints): By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

    Returns:
        ~chainer.Variable: Variable whose axes are permuted.

    """
    return Transpose(axes)(x)

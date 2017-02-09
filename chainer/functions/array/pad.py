import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Pad(function.Function):

    """Padding of an array"""

    def __init__(self, pad_width, mode, keywords):
        self.pad_width = pad_width
        self.mode = mode
        self.keywords = keywords

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_types = in_types[0]

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.tile(inputs[0], self.size, self.mode, self.keywords),

    def backward(self, inputs, grads):
        pass


def pad(x, pad_width, mode, **keywords):
    """Pad an input variable.

    Args:
        x (~chainer.Variable): Input variable to be padded.
        pad_width (int or array-like):
            Number of values padded to the edges of each axis.
        mode (str):
            'constant'
                Pads with a constant values.
        constant_values (int or array-like):
            The values are padded for each axis.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Pad(pad_width, mode, keywords)(x)

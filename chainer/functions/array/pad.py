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

        # if isinstance(pad_width, six.integer_types):
        #     self.pad_width = (pad_width,)
        # elif isinstance(pad_width, tuple) and all(
        #         isinstance(x, six.integer_types) for x in pad_width):
        #     self.pad_width = pad_width
        # elif isinstance(pad_width, tuple) and all(
        #         isinstance(x, tuple) and all(isinstance(y, six.integer_types)
        #                                      for y in x) for x in pad_width):
        #     self.pad_width = pad_width
        # else:
        #     raise TypeError('pad_width must be int, tuple of ints or tuple of tuples')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_types = in_types[0]

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.pad(inputs[0], self.pad_width, self.mode, self.keywords),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        gy = grads[0]
        # inputs = xp.array(inputs)
        for i in range(inputs.ndim):
            gy = xp.take(gy, indices=xp.arange(self.pad_width[0], self.pad_width[0] + inputs.shape[i], axis=i))
        return gy,


def pad(x, pad_width, mode, **keywords):
    """Pad an input variable.

    Args:
        x (chainer.Variable or :class:``numpy.ndarray`` or cupy.ndarray): Input data.
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

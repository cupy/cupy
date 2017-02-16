import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Pad(function.Function):

    """Padding of an array"""

    def __init__(self, pad_width, mode, **keywords):
        self.mode = mode
        self.keywords = keywords
        pad_width = numpy.asarray(pad_width)
        if pad_width.size == 1:
            pad_width = numpy.repeat(pad_width, 2)
        self.pad_width = pad_width

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        if len(self.keywords) == 0:
            return xp.pad(inputs[0], self.pad_width, mode=self.mode),
        else:
            return xp.pad(inputs[0], self.pad_width, mode=self.mode,
                          **self.keywords),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        gy = grads[0]
        array = inputs[0]
        ndims = array.ndim
        if self.pad_width.ndim == 1:
            self.pad_width = numpy.tile(self.pad_width, (ndims, 1))
        for i in range(ndims):
            gy = xp.take(gy,
                         indices=numpy.arange(self.pad_width[i][0],
                                              self.pad_width[i][0]
                                              + array.shape[i]),
                         axis=i)
        return gy,


def pad(x, pad_width, mode, **keywords):
    """Pad an input variable.

    Args:
        x (chainer.Variable or :class:``numpy.ndarray`` or cupy.ndarray):
            Input data.
        pad_width (int or array-like):
            Number of values padded to the edges of each axis.
        mode (str):
            `constant`
                Pads with a constant values.
        constant_values (int or array-like):
            The values are padded for each axis.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Pad(pad_width, mode, **keywords)(x)

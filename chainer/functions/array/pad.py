import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Pad(function.Function):

    """Padding of an array"""

    def __init__(self, pad_width, mode, **keywords):
        self.mode = mode
        self.keywords = keywords
        self.pad_width = pad_width
        self.pad_bw = numpy.asarray(pad_width)
        if self.pad_bw.size == 1:
            self.pad_bw = numpy.repeat(self.pad_bw, 2)

    def check_type_forward(self, in_types):
        # Depending on the arguments, pad_width and keywords, the input value
        # may be inappropriate. In that case, numpy.pad or cupy.pad will raise
        # errors, so that only check the size and the dtype in this function.
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.pad(inputs[0], self.pad_width, mode=self.mode,
                      **self.keywords),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        gy = grads[0]
        array = inputs[0]
        ndims = array.ndim
        if self.pad_bw.ndim == 1:
            self.pad_bw = numpy.tile(self.pad_bw, (ndims, 1))
        for i in range(ndims):
            gy = xp.take(gy,
                         indices=numpy.arange(self.pad_bw[i][0],
                                              self.pad_bw[i][0]
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
            Specifies how the function fills the periphery of the array.
            `constant`
                Pads with a constant values.
        constant_values (int or array-like):
            The values are padded for each axis.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Pad(pad_width, mode, **keywords)(x)

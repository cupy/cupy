from chainer import cuda
from chainer import function
from chainer.utils import type_check


class CReLU(function.Function):

    """Concatenated Rectified Linear Unit."""

    def __init__(self, axis=1):
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer value')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim > self.axis,
            in_types[0].ndim >= -self.axis
        )

    def get_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] *= 2
        return tuple(output_shape)

    def forward(self, x):
        x, = x
        xp = cuda.get_array_module(x)
        y = xp.empty(self.get_output_shape(x.shape), dtype=x.dtype)
        y_former, y_latter = xp.split(y, 2, axis=self.axis)
        zero = x.dtype.type(0)
        xp.maximum(zero, x, out=y_former)
        xp.maximum(zero, -x, out=y_latter)
        return y,

    def backward(self, x, gy):
        x, = x
        xp = cuda.get_array_module(x)
        gy, = gy
        gy_former, gy_latter = xp.split(gy, 2, axis=self.axis)
        return gy_former * (x > 0) - gy_latter * (-x > 0),


def crelu(x, axis=1):
    """Concatenated Rectified Linear Unit function.

    This function is expressed as :math:`f(x) = (\\max(0, x), \\max(0, -x))`,
    where two output values are concatenated along an axis.

    See: http://arxiv.org/abs/1603.05201

    Args:
        x (~chainer.Variable): Input variable.
        axis (int): Axis that the output values are concatenated along

    Returns:
        ~chainer.Variable: Output variable.

    """
    return CReLU(axis=axis)(x)

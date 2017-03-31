from chainer import cuda
from chainer import function
from chainer.utils import type_check


class FlipLR(function.Function):
    """Flip array in the left/right direction."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.fliplr(inputs[0]),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        return xp.fliplr(grads[0]),


def fliplr(a):
    """Flip array in the left/right direction.

    Args:
        xs (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return FlipLR()(a)

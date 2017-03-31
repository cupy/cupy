from chainer import cuda
from chainer import function
from chainer.utils import type_check


class FlipUD(function.Function):
    """Flip array in the up/down direction."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.flipud(inputs[0]),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        return xp.flipud(grads[0]),


def flipud(a):
    """Flip array in the up/down direction.

    Args:
        xs (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return FlipUD()(a)

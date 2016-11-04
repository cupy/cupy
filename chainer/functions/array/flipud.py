import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class FlipUD(function.Function):
    """Flip array in the up/down direction."""
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
    return FlipUD()(*xs)

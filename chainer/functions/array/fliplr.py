import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class FlipLR(function.Function):
    """Flip array in the left/right direction."""
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
    return FlipLR()(*xs)

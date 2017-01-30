from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


class Depth2Space(function.Function):

    """Depth to space transformation."""

    def __init__(self, r):
        self.r = r

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32,
                          in_types[0].ndim == 4
                          )

    def forward(self, inputs):
        X, = inputs
        xp = cuda.get_array_module(X)
        bsize, c, a, b = X.shape
        c //= self.r ** 2
        X = xp.transpose(X, (0, 2, 3, 1))
        X = xp.reshape(X, (bsize, a, b, self.r, self.r, c))
        X = xp.transpose(X, (0, 1, 3, 2, 4, 5))
        X = xp.reshape(X, (bsize, a * self.r, b * self.r, c))
        X = xp.transpose(X, (0, 3, 1, 2))
        return X,

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        xp = cuda.get_array_module(gy)
        bsize, c, a, b = gy.shape
        gy = xp.transpose(gy, (0, 2, 3, 1))
        gy = xp.reshape(gy,
                        (bsize, a // self.r, self.r, b // self.r, self.r, c)
                        )
        gy = xp.transpose(gy, (0, 1, 3, 2, 4, 5))
        gy = xp.reshape(gy,
                        (bsize, a // self.r, b // self.r, self.r ** 2 * c)
                        )
        gy = xp.transpose(gy, (0, 3, 1, 2))
        return gy,


def depth2space(X, r):
    """Computes the depth2space transformation for subpixel calculations.

    Args:
        X (Variable): Variable holding a 4d array of
        shape (batch, channel * r, dim1, dim2)
        r (int): int specifying the upscaling factor.

    Returns:
        Variable: A variable holding the upscaled array from
        interspersed depth layers.

    .. note::
       This can be used to compute super-resolution transformations.
       See https://arxiv.org/abs/1609.05158 for details.
    """
    return Depth2Space(r)(X)

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class NormalizeL2(function.Function):

    """L2 normalization"""

    def __init__(self, eps=1e-5, axis=1):
        self.eps = eps
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        norm = xp.linalg.norm(x, axis=self.axis) + self.eps
        norm = xp.expand_dims(norm, self.axis)
        return x / norm,

    def backward(self, inputs, gy):
        x = inputs[0]
        gy = gy[0]
        xp = cuda.get_array_module(x)

        norm = xp.linalg.norm(x, axis=self.axis) + self.eps
        norm = xp.expand_dims(norm, self.axis)

        x_gy_reduced = (x * gy).sum(axis=self.axis)
        x_gy_reduced = xp.expand_dims(x_gy_reduced, self.axis)
        gx = gy * norm - x_gy_reduced * x / norm
        gx = gx / norm ** 2

        return gx,


def normalize(x, eps=1e-5, axis=1):
    """L2 norm squared (a.k.a. Euclidean norm).

    This function implements L2 normalization on a vector along the given axis.
    No reduction is done along the normalization axis.

    In the case when :obj:`axis=1` and :math:`x` is a vector of dimension
    :math:`(N, K)`, where :math:`N` and :math:`K` denote mini-batch size and
    the dimension of the input variable, this function computes an output
    vector :math:`y` by the following equation:

    .. math::
       y_i = {x_i \\over \\| x_i \\|_2 + \epsilon}

    :obj:`eps` is used to avoid division by zero when norm of :math:`x` along
    the given axis is zero.

    The default value of :obj:`axis` is determined for backward compatibility.

    Args:
        x (~chainer.Variable): Two dimensional output variable. The first
            dimension is assumed to be the mini-batch dimension.
        eps (float): Epsilon value for numerical stability.
        axis (int): Axis along which to normalize.

    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.

    """
    return NormalizeL2(eps, axis)(x)

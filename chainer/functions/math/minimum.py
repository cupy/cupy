import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Minimum(function.Function):
    """Element-wise minimum of input variables."""

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 2,
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x1, x2 = inputs
        y = numpy.minimum(x1, x2)
        return utils.force_array(y),

    def backward_cpu(self, inputs, grads):
        x1, x2 = inputs
        gy, = grads
        gx1 = gy * (x1 <= x2)
        gx2 = gy * (x1 > x2)
        return utils.force_array(gx1), utils.force_array(gx2)

    def forward_gpu(self, inputs):
        x1, x2 = inputs
        return cuda.cupy.minimum(x1, x2),

    def backward_gpu(self, inputs, grads):
        x1, x2 = inputs
        gy, = grads
        gx1 = cuda.elementwise(
            'T x1, T x2, T gy', 'T gx1',
            'gx1 = (x1 <= x2) ? gy : (T)0.0',
            'minimum_bwd1')(x1, x2, gy)
        gx2 = cuda.elementwise(
            'T x1, T x2, T gy', 'T gx1',
            'gx1 = (x1 > x2) ? gy : (T)0.0',
            'minimum_bwd2')(x1, x2, gy)
        return gx1, gx2


def minimum(x1, x2):
    """Element-wise minimum of input variables.

    Args:
        x1 (~chainer.Variable): Input variables to be compared.
        x2 (~chainer.Variable): Input variables to be compared.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Minimum()(x1, x2)

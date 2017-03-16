import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class DeCov(function.Function):

    """DeCov loss (https://arxiv.org/abs/1511.06068)"""

    def __init__(self):
        self.h_centered = None
        self.covariance = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        h_type, = in_types

        type_check.expect(
            h_type.dtype == numpy.float32,
            h_type.ndim == 2,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        h, = inputs

        self.h_centered = h - h.mean(axis=0, keepdims=True)
        self.covariance = self.h_centered.T.dot(self.h_centered)
        xp.fill_diagonal(self.covariance, 0.0)
        self.covariance /= len(h)
        cost = xp.vdot(self.covariance, self.covariance) * h.dtype.type(0.5)
        return utils.force_array(cost),

    def backward(self, inputs, grad_outputs):
        h, = inputs
        gcost, = grad_outputs
        gcost_div_n = gcost / gcost.dtype.type(len(h))

        gh = 2.0 * self.h_centered.dot(self.covariance)
        gh *= gcost_div_n
        return gh,


def decov(h):
    """Computes the DeCov loss of ``h``

    Args:
        h (Variable): Variable holding a matrix where the first dimension
            corresponds to the batches.

    Returns:
        Variable: A variable holding a scalar of the DeCov loss.

    .. note::

       See https://arxiv.org/abs/1511.06068 for details.

    """
    return DeCov()(h)

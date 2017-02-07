import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class CrossCovariance(function.Function):

    """Cross-covariance loss."""

    def __init__(self):
        self.y_centered = None
        self.z_centered = None
        self.covariance = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        z_type, y_type = in_types

        type_check.expect(
            z_type.dtype == numpy.float32,
            z_type.ndim == 2,
            y_type.dtype == numpy.float32,
            y_type.ndim == 2,

            z_type.shape[0] == y_type.shape[0]
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, z = inputs

        self.y_centered = y - y.mean(axis=0, keepdims=True)
        self.z_centered = z - z.mean(axis=0, keepdims=True)
        self.covariance = self.y_centered.T.dot(self.z_centered)
        self.covariance /= len(y)
        cost = xp.vdot(self.covariance, self.covariance) * y.dtype.type(0.5)
        return utils.force_array(cost),

    def backward(self, inputs, grad_outputs):
        y, z = inputs
        gcost, = grad_outputs
        gcost_div_n = gcost / gcost.dtype.type(len(y))

        gy = self.z_centered.dot(self.covariance.T)
        gz = self.y_centered.dot(self.covariance)
        gy *= gcost_div_n
        gz *= gcost_div_n
        return gy, gz


def cross_covariance(y, z):
    """Computes the sum-squared cross-covariance penalty between ``y`` and ``z``

    Args:
        y (Variable): Variable holding a matrix where the first dimension
            corresponds to the batches.
        z (Variable): Variable holding a matrix where the first dimension
            corresponds to the batches.

    Returns:
        Variable: A variable holding a scalar of the cross covariance loss.

    .. note::

       This cost can be used to disentangle variables.
       See https://arxiv.org/abs/1412.6583v3 for details.

    """
    return CrossCovariance()(y, z)

import numpy

from chainer import function
from chainer import utils
from chainer.utils import type_check


class SquaredError(function.Function):

    """Squared error function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        return utils.force_array(self.diff * self.diff, dtype=x0.dtype),

    def backward(self, inputs, gy):
        g = gy[0] * 2 * self.diff
        return (
            utils.force_array(g, dtype=gy[0].dtype),
            utils.force_array(-g, dtype=gy[0].dtype))


def squared_error(x0, x1):
    """Squared error function.

    This function computes Squared error between two variables. Note that
    the error is not scaled by 1/2.

    """
    return SquaredError()(x0, x1)

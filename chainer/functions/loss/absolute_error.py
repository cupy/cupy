import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class AbsoluteError(function.Function):

    """Absolute error function."""

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
        return utils.force_array(abs(self.diff), dtype=x0.dtype),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        g = gy[0] * xp.sign(self.diff)
        return (
            utils.force_array(g, dtype=gy[0].dtype),
            utils.force_array(-g, dtype=gy[0].dtype))


def absolute_error(x0, x1):
    """Absolute error function.

    This function computes absolute error between two variables.

    """
    return AbsoluteError()(x0, x1)

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class HuberLoss(function.Function):

    def __init__(self, delta):
        self.delta = delta

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1 = inputs
        self.diff = x0 - x1
        y = xp.square(self.diff)
        mask = y > (self.delta ** 2)
        y -= mask * xp.square(abs(self.diff) - self.delta)
        y *= 0.5
        return y.sum(axis=1),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        mask = xp.abs(self.diff) <= self.delta
        gx = gy[0].reshape(gy[0].shape + (1,) * (self.diff.ndim - 1)) * \
            xp.where(mask, self.diff, self.delta * xp.sign(self.diff))
        return gx, -gx


def huber_loss(x, t, delta):
    """Loss function which is less sensitive to outliers in data than MSE.

        .. math::
            a = x - t

        and

        .. math::
            L_{\\delta}(a) = \\left \\{ \\begin{array}{cc}
            \\frac{1}{2} a^2 & {\\rm if~|a| \\leq \\delta} \\\\
            \\delta (|a| - \\frac{1}{2} \\delta) & {\\rm otherwise,}
            \\end{array} \\right.

    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, :math:`K`).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, :math:`K`).
        delta (float): Constant variable for huber loss function
            as used in definition.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            huber loss :math:`L_{\\delta}`.

    See:
        `Huber loss - Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`_.

    """
    return HuberLoss(delta=delta)(x, t)

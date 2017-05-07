import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class HuberLoss(function.Function):

    def __init__(self, delta, reduce='sum_along_second_axis'):
        self.delta = delta

        if reduce not in ('sum_along_second_axis', 'no'):
            raise ValueError(
                "only 'sum_along_second_axis' and 'no' are valid "
                "for 'reduce', but '%s' is given" % reduce)
        self.reduce = reduce

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
        if self.reduce == 'sum_along_second_axis':
            return y.sum(axis=1),
        else:
            return y,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        mask = xp.abs(self.diff) <= self.delta

        gx = xp.where(mask, self.diff, self.delta * xp.sign(self.diff))
        gy_ = gy[0]
        if self.reduce == 'sum_along_second_axis':
            gy_ = gy_.reshape(gy[0].shape + (1,) * (self.diff.ndim - 1))
        gx = gy_ * gx
        return gx, -gx


def huber_loss(x, t, delta, reduce='sum_along_second_axis'):
    """Loss function which is less sensitive to outliers in data than MSE.

        .. math::
            a = x - t

        and

        .. math::
            L_{\\delta}(a) = \\left \\{ \\begin{array}{cc}
            \\frac{1}{2} a^2 & {\\rm if~|a| \\leq \\delta} \\\\
            \\delta (|a| - \\frac{1}{2} \\delta) & {\\rm otherwise,}
            \\end{array} \\right.

        The output is a variable whose value depends on the value of
        the option ``reduce``. If it is ``'no'``, it holds the elementwise
        loss values. If it is ``'sum_along_second_axis'``, loss values are
        summed up along the second axis (i.e. ``axis=1``).

    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, :math:`K`).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, :math:`K`).
        delta (float): Constant variable for huber loss function
            as used in definition.
        reduce (str): Reduction option. Its value must be either
            ``'sum_along_second_axis'`` or ``'no'``. Otherwise,
            :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable object holding a scalar array of the
            huber loss :math:`L_{\\delta}`.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum_along_second_axis'``, the shape of the array
            is same as the input variables, except the second axis is removed.

    See:
        `Huber loss - Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`_.

    """
    return HuberLoss(delta=delta, reduce=reduce)(x, t)

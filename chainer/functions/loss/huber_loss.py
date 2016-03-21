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
        gx = xp.where(mask, self.diff, self.delta * xp.sign(self.diff))
        return gx, -gx


def huber_loss(x, t, delta):
    return HuberLoss(delta=delta)(x, t)

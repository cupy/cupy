import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Zoneout(function.Function):

    """Zoneout regularization."""

    def __init__(self, zoneout_ratio):
        self.zoneout_ratio = zoneout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

    def forward(self, inputs):
        h, x = inputs
        xp = cuda.get_array_module(*x)
        if xp is numpy:
            flag_x = xp.random.rand(*x.shape) >= self.zoneout_ratio
        else:
            flag_x = (xp.random.rand(*x.shape) >=
                      self.zoneout_ratio)
        self.flag_h = xp.ones_like(flag_x) ^ flag_x
        self.flag_x = flag_x
        return h * self.flag_h + x * self.flag_x,

    def backward(self, inputs, gy):
        h, x = inputs

        return gy[0] * self.flag_h, gy[0] * self.flag_x,


def zoneout(h, x, ratio=.5, train=True):
    """Drops elements of input variable and sets to previous variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    instead sets dropping element to their previous variable. In testing mode ,
    it does nothing and just returns ``x``.

    Args:
        h (~chainer.Variable): Previous variable.
        x (~chainer.Variable): Input variable.
        ratio (float): Zoneout ratio.
        train (bool): If ``True``, executes zoneout. Otherwise, return x.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper: `Zoneout: Regularizing RNNs by Randomly Preserving Hidden \
    Activations <https://arxiv.org/abs/1606.01305>`_.

    """
    if train:
        return Zoneout(ratio)(h, x)
    return x

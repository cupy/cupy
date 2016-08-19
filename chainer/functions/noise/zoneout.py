import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Zoneout(function.Function):

    """Zoneout regularization.

    See the paper: `Zoneout: Regularizing RNNs by Randomly Preserving Hidden \
    Activations <http://arxiv.org/abs/1606.013050>`_.

    """

    def __init__(self, zoneout_ratio):
        self.zoneout_ratio = zoneout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

    def forward(self, inputs):
        h, x = inputs
        xp = cuda.get_array_module(*x)
        if xp == numpy:
            flag_x = xp.random.rand(*h[0].shape) >= self.zoneout_ratio
        else:
            flag_x = (xp.random.rand(*h[0].shape, dtype=numpy.float32) >=
                      self.zoneout_ratio)
        flag_h = xp.ones_like(flag_x) - flag_x
        self.flag_h = flag_h
        self.flag_x = flag_x
        return h * self.flag_h + x * self.flag_x,

    def backward(self, inputs, gy):
        h, x = inputs

        return gy[0] * self.flag_h, gy[0] * self.flag_x,


def zoneout(h, x, ratio=.5, train=True):
    """Instead of dropping out, units zone out and are set to their previous value.

    This function stochastically forces some hidden units to maintain their
    previous values.

    Args:
        h (~chainer.Variable): Previous variable.
        x (~chainer.Variable): Input variable.
        ratio (float): Zoneout ratio.
        train (bool): If ``True``, executes zoneout. Otherwise, return x.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if train:
        return Zoneout(ratio)(h, x)
    return x

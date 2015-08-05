from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forwrad(self, in_types):
        type_check.expect(in_types.size() == 1)
        # TODO(okuta): float type check
        # type_check.expect(in_types[0].dtype == numpy.float32)

    def forward(self, x):
        scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
        xp = cuda.get_array_module(*x)
        self.mask = scale * \
            (xp.random.rand(*x[0].shape) >= self.dropout_ratio)
        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,


def dropout(x, ratio=.5, train=True):
    """Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.
        train (bool): If True, executes dropout. Otherwise, does nothing.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_.

    """
    if train:
        return Dropout(ratio)(x)
    return x

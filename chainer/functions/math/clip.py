import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Clip(function.Function):
    """Clips (limits) elements of input variable."""

    def __init__(self, x_min, xmax):
        raise NotImplementedError()


def clip(x, x_min, x_max):
    """Clips (limits) elements of input variable.

    Given an interval ``[x_min, xmax]``, elements outside the interval are
    clipped to the interval edges.

    Args:
        x (~chainer.Variable): Input variable to be clipped.
        x_min (float): Minimum value.
        x_max (float): Maximum value.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Clip(x_min, x_max)(x)

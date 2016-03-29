import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Where(function.Function):

    """Choose elements depending on condition."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        c_type, x_type, y_type = in_types

        type_check.expect(
            c_type.dtype == numpy.bool_,
            x_type.dtype == y_type.dtype,
            x_type.shape == c_type.shape,
            y_type.shape == c_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        condition, x, y = inputs
        return xp.where(condition, x, y),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        condition, x, y = inputs
        gx = xp.where(condition, grads[0], 0)
        gy = xp.where(condition, 0, grads[0])
        return None, gx, gy


def where(condition, x, y):
    """Choose elements depending on condition.

    This function choose values depending on a given ``condition``.
    All ``condition``, ``x``, and ``y`` must have the same shape.

    Args:
        condition (~chainer.Variable): Variable containing the condition.
            Only boolean array is permitted.
        x (~chainer.Variable): Variable chosen when ``condition`` is ``True``.
        y (~chainer.Variable): Variable chosen when ``condition`` is ``False``.

    Returns:
        ~chainer.Variable: Variable containing chosen values.
    """

    return Where()(condition, x, y)

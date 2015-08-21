import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Sum(function.Function):
    """Sum of array elements over a given axis."""

    def __init__(self, axis=None):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32
        )

        if self.axis is not None:
            type_check.expect(
                self.axis < in_types[0].ndim,
            )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return xp.asarray(x[0].sum(axis=self.axis)),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        if self.axis is None:
            # TODO(beam2d): Make it async
            return xp.full_like(x[0], gy[0]),

        s = list(x[0].shape)
        s[self.axis] = 1
        gy = gy[0].reshape(s)

        return xp.concatenate([gy] * x[0].shape[self.axis], axis=self.axis),


def sum(x, axis=None):
    """Sum of array elements over a given axis."""
    return Sum(axis)(x)

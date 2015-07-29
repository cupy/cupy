import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Concat(function.Function):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.Variable(self.axis, 'axis'))

        ndim = in_types[0].ndim.eval()
        for i in range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in range(0, ndim):
                if d == self.axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() > 0,
            out_types.size() == 1,
        )
        y_type, = out_types

        type_check.expect(y_type.dtype == in_types[0].dtype)
        ndim = in_types[0].ndim.eval()
        concat_size = sum(typ.shape[self.axis] for typ in in_types)
        type_check.expect(concat_size == y_type.shape[self.axis])

        for d in range(0, ndim):
            if d == self.axis:
                continue
            type_check.expect(y_type.shape[d] == in_types[0].shape[d])

    def forward(self, xs):
        return cuda.get_xpy(xs[0]).concatenate(xs, axis=self.axis),

    def backward(self, xs, gy):
        sizes = numpy.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
        return cuda.get_xpy(xs[0]).split(gy[0], sizes, axis=self.axis)


def concat(xs, axis=1):
    """Concatenates given variables along an axis.

    Args:
        xs (tuple of Variables): Variables to be concatenated.
        axis (int): Axis that the input arrays are concatenated along.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Concat(axis=axis)(*xs)

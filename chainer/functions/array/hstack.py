import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Hstack(function.Function):

    """Concatenate multiple tensors horizontally (column wise)."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = in_types[0].ndim.eval()
        for i in six.moves.range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                continue
            for d in six.moves.range(0, ndim):
                if d == 1:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)
        return xp.hstack(xs),

    def backward(self, xs, gy):
        if len(xs) == 1:
            if xs[0].ndim == 0:
                return (gy[0].reshape(()),)
            return gy

        xp = cuda.get_array_module(*xs)

        if xs[0].ndim == 0:
            ys = xp.hsplit(gy[0], len(xs))
            return [y.reshape(()) for y in ys]

        if xs[0].ndim == 1:
            sizes = numpy.array([x.shape[0] for x in xs[:-1]]).cumsum()
        else:
            sizes = numpy.array([x.shape[1] for x in xs[:-1]]).cumsum()
        return xp.hsplit(gy[0], sizes)


def hstack(xs):
    """Concatenate variables horizontally (column wise).

    Args:
        xs (list of chainer.Variable): Variables to be concatenated.

    Returns:
        ~chainer.Variable: Output variable.

    """

    return Hstack()(*xs)

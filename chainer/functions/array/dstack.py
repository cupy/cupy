import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Dstack(function.Function):

    """Concatenate multiple tensors along third axis (depth wise)."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = in_types[0].ndim.eval()
        for i in six.moves.range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 2:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in six.moves.range(0, ndim):
                if d == 2:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)
        return xp.dstack(xs),

    def backward(self, xs, gy):
        if len(xs) == 1:
            if xs[0].ndim <= 2:
                return gy[0].reshape(xs[0].shape),
            return gy

        xp = cuda.get_array_module(*xs)

        if xs[0].ndim <= 2:
            ys = xp.dsplit(gy[0], len(xs))
            return [y.reshape(xs[0].shape) for y in ys]
        else:
            sizes = numpy.array([x.shape[2] for x in xs[:-1]]).cumsum()
            return xp.dsplit(gy[0], sizes)


def dstack(xs):
    """Concatenate variables along third axis (depth wise).

    Args:
        xs (list of chainer.Variable): Variables to be concatenated.

    Returns:
        ~chainer.Variable: Output variable.

    """

    return Dstack()(*xs)

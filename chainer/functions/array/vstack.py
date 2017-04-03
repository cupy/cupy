import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Vstack(function.Function):

    """Concatenate multiple tensors vertically (row wise)."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = in_types[0].ndim.eval()
        for i in six.moves.range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in six.moves.range(1, ndim):
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        xp = cuda.get_array_module(*xs)
        return xp.vstack(xs),

    def backward(self, xs, gy):
        if len(xs) == 1:
            if xs[0].ndim <= 1:
                return gy[0].reshape(xs[0].shape),
            return gy

        xp = cuda.get_array_module(*xs)

        if xs[0].ndim <= 1:
            ys = xp.vsplit(gy[0], len(xs))
            return [y.reshape(xs[0].shape) for y in ys]
        else:
            sizes = numpy.array([x.shape[0] for x in xs[:-1]]).cumsum()
            return xp.vsplit(gy[0], sizes)


def vstack(xs):
    """Concatenate variables vertically (row wise).

    Args:
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variables to be concatenated. The variables must have the
            same shape, except in the vertical (row wise) dimension.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x1 = np.arange(0, 12).reshape(3, 4)
        >>> x1
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> x2 = np.arange(12, 20).reshape(2, 4)
        >>> x2
        array([[12, 13, 14, 15],
               [16, 17, 18, 19]])
        >>> F.vstack([x1, x2]).data
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11],
               [12, 13, 14, 15],
               [16, 17, 18, 19]])

    """

    return Vstack()(*xs)

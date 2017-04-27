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
            same ``ndim``. When the variables have the second axis (i.e.
            :math:`ndim \\geq 2`), the variables must have the same shape
            along all but the first axis. When the variables do not have the
            second axis(i.e. :math:`ndim < 2`), the variables must have the
            same shape.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x1 = np.array((1, 2, 3))
        >>> x1.shape
        (3,)
        >>> x2 = np.array((2, 3, 4))
        >>> x2.shape
        (3,)
        >>> y = F.vstack((x1, x2))
        >>> y.shape
        (2, 3)
        >>> y.data
        array([[1, 2, 3],
               [2, 3, 4]])
        >>> x1 = np.arange(0, 12).reshape(3, 4)
        >>> x1.shape
        (3, 4)
        >>> x1
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> x2 = np.arange(12, 20).reshape(2, 4)
        >>> x2.shape
        (2, 4)
        >>> x2
        array([[12, 13, 14, 15],
               [16, 17, 18, 19]])
        >>> y = F.vstack([x1, x2])
        >>> y.shape
        (5, 4)
        >>> y.data
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11],
               [12, 13, 14, 15],
               [16, 17, 18, 19]])

    """

    return Vstack()(*xs)

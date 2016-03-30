import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class BinaryAccuracy(function.Function):

    ignore_label = -1

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            t_type.shape == x_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs
        # flatten
        y = y.ravel()
        t = t.ravel()
        c = (y >= 0)
        count = xp.maximum(1, (t != self.ignore_label).sum())
        return xp.asarray((c == t).sum(dtype='f') / count, dtype='f'),


def binary_accuracy(y, t):
    """Computes binary classification accuracy of the minibatch.

    Args:
        y (Variable): Variable holding a matrix whose i-th element
            indicates the score of positive at the i-th example.
        t (Variable): Variable holding an int32 vector of ground truth labels.
            If ``t[i] == -1``, corresponding ``x[i]`` is ignored.
            Accuracy is zero if all ground truth labels are ``-1``.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    """
    return BinaryAccuracy()(y, t)

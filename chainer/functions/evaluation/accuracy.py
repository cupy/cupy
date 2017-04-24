import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Accuracy(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32
        )

        t_ndim = t_type.ndim.eval()
        type_check.expect(
            x_type.ndim >= t_type.ndim,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
        )
        for i in six.moves.range(t_ndim + 1, x_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            ignore_cnt = mask.sum()

            # will always be true when the true label is ignore_label
            # TODO(henry0312)
            #   If cupy.where returns indexes, we could make the code better.
            #   Also, we would need Advanced Indexing.
            pred = xp.where(mask, self.ignore_label,
                            y.argmax(axis=1).reshape(t.shape))
            count = (pred == t).sum() - ignore_cnt
            total = t.size - ignore_cnt

            if total == 0:
                return xp.asarray(0.0, dtype=y.dtype),
            else:
                return xp.asarray(float(count) / total, dtype=y.dtype),
        else:
            pred = y.argmax(axis=1).reshape(t.shape)
            return xp.asarray((pred == t).mean(dtype=y.dtype)),


def accuracy(y, t, ignore_label=None):
    """Computes multiclass classification accuracy of the minibatch.

    Args:
        y (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Array whose (i, j, k, ...)-th element indicates the score of
            the class j at the (i, k, ...)-th sample.
            The prediction label :math:`\\hat t` is calculated by the formula
            :math:`\\hat t_(i, k, ...) = argmax_j y(i, j, k, ...)`.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray` of :class:`numpy.int32`):
            Array of ground truth labels.
        ignore_label (int or None): Skip calculating accuracy
            if the true label is ``ignore_label``.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    .. admonition:: Example

        We show the most common case, when ``y`` is the two dimensional array.

        >>> y = np.array([[0.1, 0.7, 0.2], # prediction label is 1
        ...               [8.0, 1.0, 2.0], # prediction label is 0
        ...               [-8.0, 1.0, 2.0], # prediction label is 2
        ...               [-8.0, -1.0, -2.0]]) # prediction label is 1
        >>> t = np.array([1, 0, 2, 1], 'i')
        >>> F.accuracy(y, t).data \
# 100% accuracy because all samples are correct
        array(1.0)
        >>> t = np.array([1, 0, 0, 0], 'i')
        >>> F.accuracy(y, t).data \
# 50% accuracy because 1st and 2nd samples are correct.
        array(0.5)
        >>> F.accuracy(y, t, ignore_label=0).data \
# 100% accuracy because of ignoring the 2nd, 3rd and 4th samples.
        array(1.0)

    """
    return Accuracy(ignore_label=ignore_label)(y, t)

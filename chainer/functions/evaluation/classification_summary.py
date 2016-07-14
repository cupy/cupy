from __future__ import division

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _f1_score(precision, recall, beta):
    beta_square = beta * beta
    return ((1 + beta_square) * precision * recall /
            (beta_square * precision + recall)).astype(precision.dtype)


class ClassificationSummary(function.Function):

    ignore_label = -1

    def __init__(self, label_num, beta):
        self.label_num = label_num
        self.beta = beta

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

        mask = (t == self.ignore_label)
        pred = xp.where(mask, self.ignore_label,
                        y.argmax(axis=1).reshape(t.shape))

        if self.label_num is None:
            label_num = xp.maximum(t) + 1
        else:
            label_num = self.label_num
            if chainer.is_debug():
                assert (t < label_num).all()

        precision = xp.empty((label_num,), dtype=y.dtype)
        recall = xp.empty((label_num), dtype=y.dtype)
        support = xp.empty((label_num), dtype=numpy.int32)

        # TODO(Kenta Oono)
        # speed up
        for i in six.moves.range(label_num):
            supp = (t == i) & (t != self.ignore_label)
            relevant = (pred == i) & (t != self.ignore_label)
            tp = (supp & relevant).sum().astype(y.dtype)
            support[i] = supp.sum()
            precision[i] = tp / relevant.sum()
            recall[i] = tp / support[i]

        f1 = _f1_score(precision, recall, self.beta)

        return precision, recall, f1, support


def classification_summary(y, t, label_num=None, beta=1.0):
    """Calculates Precision, Recall, F1 Score, and support.

    This function calculates the following quantities for each class.

    - Precision: :math:`\frac{tp}{tp + fp}`
    - Recall: :math:`\frac{tp}{tp + tn}`
    - F beta Score: The weighted harmonic average of Precision and Recall.
    - Support: The number of instances of each ground truth label.

    Here, ``tp``, ``fp``, and ``tn`` stand for the number of true positives,
    false positives, and true negative, respectively.

    If ``ignore_label`` is not ``None``, instances with the given
    label are ignored.
    ``label_num`` specifies the number of classes, that is,
    each value in ``t`` must be an integer in the range of
    ``[0, label_num)``.
    If ``label_num`` is ``None``, this function regards
    ``label_num`` as a maximum of in ``t`` plus one.

    Args:
        y (~chainer.Variable): Variable holding a vector of scores.
        t (~chainer.Variable): Variable holding a vector of
        ground truth labels.
        label_num (int): The number of classes.
        beta (float): The parameter which determines the weight of
        precision in the F-beta score.

    Returns:
        4-tuple of ~chainer.Variable of size ``(label_num,)``.
        Each element represents precision, recall, F beta score,
        and support of this minibatch.

    """
    return ClassificationSummary(label_num, beta)(y, t)


def precision(y, t, label_num=None):
    ret = ClassificationSummary(label_num, 1.0)(y, t)
    return ret[0], ret[-1]


def recall(y, t, label_num=None):
    ret = ClassificationSummary(label_num, 1.0)(y, t)
    return ret[1], ret[-1]


def f1_score(y, t, label_num=None, beta=1.0):
    ret = ClassificationSummary(beta, label_num)(y, t)
    return ret[2], ret[-1]

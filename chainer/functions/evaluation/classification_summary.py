from __future__ import division

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _fbeta_score(precision, recall, beta):
    beta_square = beta * beta
    return ((1 + beta_square) * precision * recall /
            (beta_square * precision + recall)).astype(precision.dtype)


class ClassificationSummary(function.Function):

    def __init__(self, label_num, beta, ignore_label):
        self.label_num = label_num
        self.beta = beta
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

        if self.label_num is None:
            label_num = xp.amax(t) + 1
        else:
            label_num = self.label_num
            if chainer.is_debug():
                assert (t < label_num).all()

        mask = (t == self.ignore_label).ravel()
        pred = xp.where(mask, label_num, y.argmax(axis=1).ravel())
        true = xp.where(mask, label_num, t.ravel())
        support = xp.bincount(true, minlength=label_num + 1)[:label_num]
        relevant = xp.bincount(pred, minlength=label_num + 1)[:label_num]
        tp_mask = xp.where(pred == true, true, label_num)
        tp = xp.bincount(tp_mask, minlength=label_num + 1)[:label_num]

        precision = tp / relevant
        recall = tp / support
        fbeta = _fbeta_score(precision, recall, self.beta)

        return precision, recall, fbeta, support


def classification_summary(y, t, label_num=None, beta=1.0, ignore_label=-1):
    """Calculates Precision, Recall, F beta Score, and support.

    This function calculates the following quantities for each class.

    - Precision: :math:`\\frac{\\mathrm{tp}}{\\mathrm{tp} + \\mathrm{fp}}`
    - Recall: :math:`\\frac{\\mathrm{tp}}{\\mathrm{tp} + \\mathrm{tn}}`
    - F beta Score: The weighted harmonic average of Precision and Recall.
    - Support: The number of instances of each ground truth label.

    Here, ``tp``, ``fp``, and ``tn`` stand for the number of true positives,
    false positives, and true negative, respectively.

    ``label_num`` specifies the number of classes, that is,
    each value in ``t`` must be an integer in the range of
    ``[0, label_num)``.
    If ``label_num`` is ``None``, this function regards
    ``label_num`` as a maximum of in ``t`` plus one.

    ``ignore_label`` determines which instances should be ignored.
    Specifically, instances with the given label are not taken
    into account for calculating the above quantities.
    By default, it is set to -1 so that all instances are taken
    into consideration, as labels are supposed to be non-negative integers.
    Setting ``ignore_label`` to a non-negative integer less than ``label_num``
    is illegal and yields undefined behavior. In the current implementation,
    it arises ``RuntimeWarning`` and ``ignore_label``-th entries in output
    arrays do not contain correct quantities.

    Args:
        y (~chainer.Variable): Variable holding a vector of scores.
        t (~chainer.Variable): Variable holding a vector of
            ground truth labels.
        label_num (int): The number of classes.
        beta (float): The parameter which determines the weight of
            precision in the F-beta score.
        ignore_label (int): Instances with this label are ignored.

    Returns:
        4-tuple of ~chainer.Variable of size ``(label_num,)``.
        Each element represents precision, recall, F beta score,
        and support of this minibatch.

    """
    return ClassificationSummary(label_num, beta, ignore_label)(y, t)


def precision(y, t, label_num=None, ignore_label=-1):
    ret = ClassificationSummary(label_num, 1.0, ignore_label)(y, t)
    return ret[0], ret[-1]


def recall(y, t, label_num=None, ignore_label=-1):
    ret = ClassificationSummary(label_num, 1.0, ignore_label)(y, t)
    return ret[1], ret[-1]


def fbeta_score(y, t, label_num=None, beta=1.0, ignore_label=-1):
    ret = ClassificationSummary(label_num, beta, ignore_label)(y, t)
    return ret[2], ret[-1]


def f1_score(y, t, label_num=None, ignore_label=-1):
    ret = ClassificationSummary(label_num, 1.0, ignore_label)(y, t)
    return ret[2], ret[-1]

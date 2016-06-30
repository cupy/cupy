from __future__ import division

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class F1Score(function.Function):

    def __init__(self, beta, positive_label):
        self.beta = beta
        self.positive_label = positive_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32,
            t_type.shape == x_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        positive = t == self.positive_label
        relevant = y >= 0
        tp = (relevant & positive).sum()
        self.precision = xp.asarray(tp / relevant.sum(), dtype=y.dtype)
        self.recall = xp.asarray(tp / positive.sum(), dtype=y.dtype)
        beta_square = self.beta * self.beta
        self.f1_score = xp.asarray((1 + beta_square) * self.precision * self.recall
                                   / (beta_square * self.precision + self.recall), dtype=y.dtype)
        return self.f1_score,


def f1_score(y, t, beta=1.0, positive_label=0):
    return F1Score(beta, positive_label)(y, t)

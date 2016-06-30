import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


def recall(ys, ts, dtype):
    tp = 0
    positive = 0
    for y, t in zip(ys.ravel(), ts.ravel()):
        if y >= 0 and t == 0:
            tp += 1
        if t == 0:
            positive += 1
    return dtype(tp) / positive
            

def precision(ys, ts, dtype):
    tp = 0
    relevant = 0
    for y, t in zip(ys.ravel(), ts.ravel()):
        if y >= 0 and t == 0:
            tp += 1
        if y >= 0:
            relevant += 1
    return dtype(tp) / relevant


def f1_score(precision, recall, beta=1.0):
    beta_square = beta * beta
    return (1 + beta_square) * precision * recall / (beta_square * precision + recall)


@testing.parameterize(
    *testing.product({
        'x_shape': [(10, 3), (10, 3, 5)],
        'dtype': [numpy.float16, numpy.float32, numpy.float64]}
    )
)
class TestF1Score(unittest.TestCase):

    def setUp(self):
        self.y = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.t = numpy.random.randint(0, 2, self.x_shape).astype(numpy.int32)
        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, xp):
        y = chainer.Variable(xp.asarray(self.y))
        t = chainer.Variable(xp.asarray(self.t))
        f1_actual = F.f1_score(y, t)
        p = precision(self.y, self.t, self.dtype)
        t = recall(self.y, self.t, self.dtype)
        f1_expect = f1_score(p, t)
        chainer.testing.assert_allclose(f1_actual.data, f1_expect,
                                        **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)

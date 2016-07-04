import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def recall(preds, ts, dtype, label_num):
    tp = numpy.zeros((label_num,), dtype=numpy.int32)
    support = numpy.zeros((label_num,), dtype=numpy.int32)
    for p, t in zip(preds.ravel(), ts.ravel()):
        support[t] += 1
        if p == t:
            tp[t] += 1
    return dtype(tp) / support


def precision(preds, ts, dtype, label_num):
    tp = numpy.zeros((label_num,), dtype=numpy.int32)
    relevant = numpy.zeros((label_num,), dtype=numpy.int32)
    for p, t in zip(preds.ravel(), ts.ravel()):
        relevant[p] += 1
        if p == t:
            tp[p] += 1
    return dtype(tp) / relevant


def f1_score(precision, recall, beta=1.0):
    beta_square = beta * beta
    return ((1 + beta_square) * precision * recall /
            (beta_square * precision + recall))


def support(ts, dtype, label_num):
    ret = numpy.zeros((label_num,), dtype=numpy.int32)
    for t in ts.ravel():
        ret[t] += 1
    return ret


@testing.parameterize(
    *testing.product_dict(
        [{'y_shape': (30, 3), 't_shape': (30,)},
         {'y_shape': (30, 3, 5), 't_shape': (30, 5)}],
        [{'dtype': numpy.float16},
         {'dtype': numpy.float32},
         {'dtype': numpy.float64}],
        [{'beta': 1.0},
         {'beta': 2.0}]
    )
)
class TestClassificationSummary(unittest.TestCase):

    def setUp(self):
        self.label_num = 3
        self.y = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)
        self.t = numpy.random.randint(
            0, self.label_num, self.t_shape).astype(numpy.int32)
        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, xp):
        y = chainer.Variable(xp.asarray(self.y))
        t = chainer.Variable(xp.asarray(self.t))
        p_actual, r_actual, f1_actual, s_actual = F.classification_summary(
            y, t, self.label_num, self.beta)

        pred = self.y.argmax(axis=1).reshape(self.t.shape)
        p_expect = precision(pred, self.t, self.dtype, self.label_num)
        r_expect = recall(pred, self.t, self.dtype, self.label_num)
        f1_expect = f1_score(p_expect, r_expect, self.beta)
        s_expect = support(self.t, self.dtype, self.label_num)
        chainer.testing.assert_allclose(p_actual.data, p_expect,
                                        **self.check_forward_options)
        chainer.testing.assert_allclose(r_actual.data, r_expect,
                                        **self.check_forward_options)
        chainer.testing.assert_allclose(f1_actual.data, f1_expect,
                                        **self.check_forward_options)
        chainer.testing.assert_allclose(s_actual.data, s_expect,
                                        **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(numpy)

    @condition.retry(3)
    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)


testing.run_module(__name__, __file__)

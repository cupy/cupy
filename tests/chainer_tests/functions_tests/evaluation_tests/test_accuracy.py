import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(10, 3), (20, 3)],
    't_data': ['random', 'zero'],
    'ignore_label': [None, 0],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestAccuracy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.t_data == 'random':
            self.t = numpy.random.randint(
                3, size=self.shape[:1]).astype(numpy.int32)
        elif self.t_data == 'zero':
            self.t = numpy.zeros(self.shape[:1], dtype=numpy.int32)

        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.accuracy(x, t, self.ignore_label)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual((), y.data.shape)

        if self.ignore_label is not None:
            count = 0
            for i in six.moves.range(self.t.size):
                pred = self.x[i].argmax()
                if self.t[i] != self.ignore_label and pred == self.t[i]:
                    count += 1
                total = (self.t != self.ignore_label).sum()
        else:
            count = 0
            for i in six.moves.range(self.t.size):
                pred = self.x[i].argmax()
                if pred == self.t[i]:
                    count += 1
                total = self.t.size

        if total == 0:
            expected = 0.0
        else:
            expected = float(count) / total
        gradient_check.assert_allclose(
            expected, cuda.to_cpu(y.data), **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


testing.run_module(__name__, __file__)

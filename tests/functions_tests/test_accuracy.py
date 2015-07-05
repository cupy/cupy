import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestAccuracy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=(10,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.accuracy(x, t)
        self.assertEqual((), y.data.shape)

        count = 0
        for i in six.moves.range(self.t.size):
            pred = self.x[i].argmax()
            if pred == self.t[i]:
                count += 1

        expected = float(count) / self.t.size
        gradient_check.assert_allclose(expected, cuda.to_cpu(y.data))

    @condition.success_at_least(3, 1)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.success_at_least(3, 1)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

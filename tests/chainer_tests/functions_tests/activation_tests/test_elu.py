import random
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestELU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numeraical grad
        self.x = numpy.random.uniform(-1, 1, (5, 4)).astype(numpy.float32)
        for i in range(self.x.size):
            if -0.01 < self.x.flat[i] < 0.01:
                self.x.flat[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, (5, 4)).astype(numpy.float32)
        self.alpha = random.random()

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.elu(x, alpha=self.alpha)
        self.assertEqual(y.data.dtype, numpy.float32)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] = self.alpha * (numpy.exp(expected[i]) - 1)

        gradient_check.assert_allclose(expected, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.ELU(self.alpha), x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

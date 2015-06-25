from unittest import TestCase

import numpy

from chainer import cuda
from chainer.cuda import to_cpu
from chainer.cuda import to_gpu
from chainer.functions import mean_squared_error
from chainer.gradient_check import assert_allclose
from chainer.gradient_check import numerical_grad
from chainer.testing import attr
from chainer import Variable


if cuda.available:
    cuda.init()


class TestMeanSquaredError(TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_forward(self, x0_data, x1_data):
        x0 = Variable(x0_data)
        x1 = Variable(x1_data)
        loss = mean_squared_error(x0, x1)
        loss_value = float(to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.
        for i in numpy.ndindex(self.x0.shape):
            loss_expect += (self.x0[i] - self.x1[i]) ** 2
        loss_expect /= self.x0.size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    def test_forwrad_gpu(self):
        self.check_forward(to_gpu(self.x0), to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data):
        x0 = Variable(x0_data)
        x1 = Variable(x1_data)
        loss = mean_squared_error(x0, x1)
        loss.backward()

        func = loss.creator
        f = lambda: func.forward((x0.data, x1.data))
        gx0, gx1 = numerical_grad(f, (x0.data, x1.data), (1,), eps=1e-2)

        assert_allclose(gx0, x0.grad)
        assert_allclose(gx1, x1.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x0), to_gpu(self.x1))

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestNonparameterizedLinear(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, (2, 3)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            -1, 1, 2).astype(numpy.float32)

        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)
        self.y = self.x.dot(self.W.T) + self.b

    def check_forward(self, x_data, W_data, b_data, y_expect):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)
        y = functions.linear(x, W, b)
        gradient_check.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b,
                           self.x.dot(self.W.T) + self.b)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b),
            cuda.to_gpu(self.x.dot(self.W.T) + self.b))

    def check_backward(self, x_data, W_data, b_data, y_grad):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)
        y = functions.linear(x, W, b)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, W.data, b.data))
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, W.data, b.data), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, W.grad)
        gradient_check.assert_allclose(gb, b.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

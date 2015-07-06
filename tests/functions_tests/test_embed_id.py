import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestEmbedID(unittest.TestCase):

    def setUp(self):
        self.func = functions.EmbedID(3, 2)
        self.func.gW.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU
        self.x = numpy.array([0, 1, 0], dtype=numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def to_gpu(self):
        self.func.W = cuda.to_gpu(self.func.W)
        self.func.gW = cuda.to_gpu(self.func.gW)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.empty_like(self.gy)
        for i in six.moves.range(self.x.size):
            y_expect[i] = self.W[int(self.x[i])]

        gradient_check.assert_allclose(y_expect, y.data, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gW, = gradient_check.numerical_grad(f, (func.W,), (y.grad,))
        gradient_check.assert_allclose(gW, func.gW)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

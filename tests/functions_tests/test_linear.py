import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestLinear(unittest.TestCase):

    in_shape = (3,)
    out_size = 2

    def setUp(self):
        in_size = numpy.prod(self.in_shape)
        self.func = functions.Linear(in_size, self.out_size)
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.b = numpy.random.uniform(
            -1, 1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU
        self.b = self.func.b.copy()  # fixed on CPU

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y = self.x.reshape(4, -1).dot(self.func.W.T) + self.func.b

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        gradient_check.assert_allclose(self.y, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.func.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, func.W, func.b), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, func.gW)
        gradient_check.assert_allclose(gb, func.gb)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestLinearWithSpatialDimensions(TestLinear):

    in_shape = (3, 2, 2)

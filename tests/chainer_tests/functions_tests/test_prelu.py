import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestPReLUSingle(unittest.TestCase):

    def setUp(self):
        self.func = functions.PReLU()
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.gW.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU

        # Avoid unstability of numerical gradient
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        for i in range(self.x.size):
            if -0.01 < self.x.flat[i] < 0.01:
                self.x.flat[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                y_expect[i] *= self.W

        gradient_check.assert_allclose(y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
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
        gx, gW = gradient_check.numerical_grad(f, (x.data, func.W), (y.grad,))

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, func.gW, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestPReLUMulti(TestPReLUSingle):

    def setUp(self):
        self.func = functions.PReLU(shape=(3,))
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.gW.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU

        # Avoid unstability of numerical gradient
        self.x = numpy.random.uniform(.5, 1, (4, 3, 2)).astype(numpy.float32)
        self.x *= numpy.random.randint(2, size=(4, 3, 2)) * 2 - 1
        self.gy = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                y_expect[i] *= self.W[i[1]]

        gradient_check.assert_allclose(y_expect, y.data)


testing.run_module(__name__, __file__)

from unittest import TestCase

import numpy
from pycuda.gpuarray import to_gpu

from chainer import Variable
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import sum

class TestSum(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.array([2], dtype=numpy.float32)

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = sum(x)
        y_expect = self.x.sum()
        assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = sum(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

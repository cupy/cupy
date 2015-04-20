from unittest import TestCase

import numpy
from pycuda.gpuarray import to_gpu

from chainer import Variable
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import tanh

class TestSigmoid(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-.5, .5, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    def test_forward_gpu(self):
        x = Variable(to_gpu(self.x))
        y = tanh(x)
        y_expect = tanh(Variable(self.x))
        assert_allclose(y_expect.data, y.data)

    def check_backward(self, x_data, gy_data):
        x = Variable(x_data)
        y = tanh(x)
        y.grad = gy_data
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))
        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

from unittest import TestCase

import numpy
from pycuda.gpuarray import to_gpu

from chainer import Variable
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import Convolution2D

class TestConvolution2D(TestCase):
    def setUp(self):
        self.func = Convolution2D(3, 2, 3, stride=2, pad=1)
        self.func.b = numpy.random.uniform(
            -1, 1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)

        self.x  = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 2, 2, 2)).astype(numpy.float32)

    def to_gpu(self):
        self.func.W  = to_gpu(self.func.W)
        self.func.b  = to_gpu(self.func.b)
        self.func.gW = to_gpu(self.func.gW)
        self.func.gb = to_gpu(self.func.gb)

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = numerical_grad(f, (x.data, func.W, func.b), (y.grad,), eps=1e-2)

        assert_allclose(gx, x.grad)
        assert_allclose(gW, func.gW)
        assert_allclose(gb, func.gb)

    def test_backward_gpu(self):
        self.to_gpu()
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

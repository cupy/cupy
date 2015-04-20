from unittest import TestCase

import numpy
from pycuda import gpuarray

from chainer import Variable
from chainer.gradient_check import numerical_grad, l_infty_dist
from chainer.functions import Convolution2D

class TestConvolution2D(TestCase):
    def setUp(self):
        self.func = Convolution2D(3, 2, 3, stride=2, pad=1)
        self.func.b = numpy.random.uniform(
            -.1, .1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)

        self.x  = numpy.random.uniform(-.5, .5, (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (2, 2, 2, 2)).astype(numpy.float32)

    def to_gpu(self):
        self.func.W  = gpuarray.to_gpu(self.func.W)
        self.func.b  = gpuarray.to_gpu(self.func.b)
        self.func.gW = gpuarray.to_gpu(self.func.gW)
        self.func.gb = gpuarray.to_gpu(self.func.gb)

    def check_backward(self, x, y):
        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = numerical_grad(f, (x.data, func.W, func.b), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)
        self.assertLess(l_infty_dist(gW, func.gW), 1e-5)
        self.assertLess(l_infty_dist(gb, func.gb), 1e-5)

    def test_backward_gpu(self):
        self.to_gpu()
        x = Variable(gpuarray.to_gpu(self.x))
        y = self.func(x)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()
        self.check_backward(x, y)

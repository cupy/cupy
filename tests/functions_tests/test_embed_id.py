from unittest import TestCase

import numpy
import pycuda.gpuarray as gpuarray

from chainer import Variable
from chainer.gradient_check import numerical_grad, l_infty_dist
from chainer.functions import EmbedID

class TestEmbedID(TestCase):
    def setUp(self):
        self.func = EmbedID(3, 2)
        self.func.gW.fill(0)

        self.W  = self.func.W.copy()  # fixed on CPU
        self.x  = numpy.array([0, 1, 0], dtype=numpy.int32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    def to_gpu(self):
        self.func.W  = gpuarray.to_gpu(self.func.W)
        self.func.gW = gpuarray.to_gpu(self.func.gW)

    def compute_expected_y(self):
        y = numpy.empty_like(self.gy)
        for i in xrange(self.x.size):
            y[i] = self.W[int(self.x[i])]
        return y

    def test_forward_cpu(self):
        x = Variable(self.x)
        y = self.func(x)
        y_expect = self.compute_expected_y()
        self.assertTrue((y_expect == y.data).all())

    def test_forward_gpu(self):
        self.to_gpu()
        x = Variable(gpuarray.to_gpu(self.x))
        y = self.func(x)
        y_expect = self.compute_expected_y()
        self.assertTrue((y_expect == y.data.get()).all())

    def check_backward(self, x, y):
        func = y.creator
        f = lambda: func.forward((x.data,))
        gW, = numerical_grad(f, (func.W,), (y.grad,))

        self.assertLess(l_infty_dist(gW, func.gW), 1e-5)

    def test_backward_cpu(self):
        x = Variable(self.x)
        y = self.func(x)
        y.grad = self.gy
        y.backward()
        self.check_backward(x, y)

    def test_backward_gpu(self):
        self.to_gpu()
        x = Variable(gpuarray.to_gpu(self.x))
        y = self.func(x)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()
        self.check_backward(x, y)

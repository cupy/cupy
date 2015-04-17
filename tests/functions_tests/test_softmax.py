from unittest import TestCase

import numpy
import pycuda.gpuarray as gpuarray

from chainer import Variable
from chainer.gradient_check import numerical_grad, l_infty_dist
from chainer.functions import softmax

class TestSoftmax(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-.5, .5, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (2, 3)).astype(numpy.float32)

        self.y_expect = numpy.exp(self.x)
        for i in xrange(self.y_expect.shape[0]):
            self.y_expect[i] /= self.y_expect[i].sum()

    def test_forward_cpu(self):
        x = Variable(self.x)
        y = softmax(x)
        self.assertLess(l_infty_dist(self.y_expect, y.data), 1e-7)

    def test_forward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = softmax(x)
        self.assertLess(l_infty_dist(self.y_expect, y.data.get()), 1e-7)

    def test_backward_cpu(self):
        x = Variable(self.x)
        y = softmax(x)
        y.grad = self.gy
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

    def test_backward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = softmax(x)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()

        x2 = Variable(self.x)
        y2 = softmax(x2)
        y2.grad = self.gy
        y2.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

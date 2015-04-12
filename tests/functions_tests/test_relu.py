from unittest import TestCase

import numpy
import pycuda.gpuarray as gpuarray

from chain import Variable
from chain.gradient_check import numerical_grad, l_infty_dist
from chain.functions import relu

class TestReLU(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-.5, .5, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    def test_backward_cpu(self):
        x = Variable(self.x)
        y = relu(x)
        y.grad = self.gy
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

    def test_backward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = relu(x)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

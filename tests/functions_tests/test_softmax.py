from unittest import TestCase

import numpy
import pycuda.gpuarray as gpuarray

from chain import Variable
from chain.gradient_check import numerical_grad, l_infty_dist
from chain.functions import softmax

class TestSoftmax(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-.5, .5, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (2, 3)).astype(numpy.float32)

    def test_forward_cpu(self):
        x = Variable(self.x)
        y = softmax(x)
        y_expect = numpy.exp(x.data)
        for i in xrange(y_expect.shape[0]):
            y_expect[i] /= y_expect[i].sum()
        self.assertLess(l_infty_dist(y_expect, y.data), 1e-7)

    def test_backward_cpu(self):
        x = Variable(self.x)
        y = softmax(x)
        y.grad = self.gy
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

from unittest import TestCase

import numpy
from pycuda import gpuarray

from chainer import Variable
from chainer.gradient_check import numerical_grad, l_infty_dist
from chainer.functions import Linear

class TestLinear(TestCase):
    def setUp(self):
        self.fgen = Linear(3, 2)
        self.fgen.b = numpy.random.uniform(
            -.1, .1, self.fgen.b.shape).astype(numpy.float32)
        self.fgen.gW.fill(0)
        self.fgen.gb.fill(0)

        self.x    = numpy.random.uniform(-.5, .5, (4, 3)).astype(numpy.float32)
        self.gy   = numpy.random.uniform(-.1, .1, (4, 2)).astype(numpy.float32)
        self.y    = self.x.dot(self.fgen.W.T) + self.fgen.b

    def to_gpu(self):
        self.fgen.W = gpuarray.to_gpu(self.fgen.W)
        self.fgen.b = gpuarray.to_gpu(self.fgen.b)
        self.fgen.gW = gpuarray.to_gpu(self.fgen.gW)
        self.fgen.gb = gpuarray.to_gpu(self.fgen.gb)

    def test_forward_cpu(self):
        x = Variable(self.x)
        y = self.fgen(x)
        self.assertTrue((self.y == y.data).all())

    def test_forward_gpu(self):
        self.to_gpu()
        x = Variable(gpuarray.to_gpu(self.x))
        y = self.fgen(x)
        self.assertLess(l_infty_dist(self.y, y.data.get()), 1e-6)

    def check_backward(self, x, y):
        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = numerical_grad(f, (x.data, func.W, func.b), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)
        self.assertLess(l_infty_dist(gW, func.gW), 1e-5)
        self.assertLess(l_infty_dist(gb, func.gb), 1e-5)

    def test_backward_cpu(self):
        x = Variable(self.x)
        y = self.fgen(x)
        y.grad = self.gy
        y.backward()
        self.check_backward(x, y)

    def test_backward_gpu(self):
        self.to_gpu()
        x = Variable(gpuarray.to_gpu(self.x))
        y = self.fgen(x)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()
        self.check_backward(x, y)

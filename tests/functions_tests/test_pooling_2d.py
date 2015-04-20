from unittest import TestCase

import numpy
from numpy.testing import assert_allclose
from pycuda import gpuarray

from chainer import Variable
from chainer.gradient_check import numerical_grad, l_infty_dist
from chainer.functions import average_pooling_2d, max_pooling_2d

class TestMaxPooling2D(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3, 2, 2)).astype(numpy.float32)

    def check_forward(self, y):
        self.assertEqual((2, 3, 2, 2), y.shape)
        for k in xrange(2):
            for c in xrange(3):
                expect = numpy.array([
                    [self.x[k, c, 0:2, 0:2].max(), self.x[k, c, 0:2, 1:3].max()],
                    [self.x[k, c, 1:4, 0:2].max(), self.x[k, c, 1:4, 1:3].max()]])
                assert_allclose(expect, y[k, c])

    def test_forward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = max_pooling_2d(x, 3, stride=2, pad=1)
        self.check_forward(y.data.get())

    def check_backward(self, x, y):
        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))
        assert_allclose(gx.get(), x.grad.get(), atol=1e-5, rtol=1e-4)

    def test_backward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = max_pooling_2d(x, 3, stride=2, pad=1)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()
        self.check_backward(x, y)


class TestAveragePooling2D(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3, 2, 2)).astype(numpy.float32)

    def check_forward(self, y):
        self.assertEqual((2, 3, 2, 2), y.shape)
        for k in xrange(2):
            for c in xrange(3):
                expect = numpy.array([
                    [self.x[k, c, 0:2, 0:2].sum(), self.x[k, c, 0:2, 1:3].sum()],
                    [self.x[k, c, 1:4, 0:2].sum(), self.x[k, c, 1:4, 1:3].sum()]]
                ) / 9
                assert_allclose(expect, y[k, c])

    def test_forward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = average_pooling_2d(x, 3, stride=2, pad=1)
        self.check_forward(y.data.get())

    def check_backward(self, x, y):
        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))
        assert_allclose(gx.get(), x.grad.get(), atol=1e-5, rtol=1e-4)

    def test_backward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        y = average_pooling_2d(x, 3, stride=2, pad=1)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()
        self.check_backward(x, y)

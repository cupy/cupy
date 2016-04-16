import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestLinearInterpolate(unittest.TestCase):

    shape = (3, 4)

    def setUp(self):
        self.p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, p_data, x_data, y_data):
        p = chainer.Variable(p_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        z = functions.linear_interpolate(p, x, y)
        expect = self.p * self.x + (1 - self.p) * self.y
        gradient_check.assert_allclose(z.data, expect)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.p, self.x, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.p),
                           cuda.to_gpu(self.x),
                           cuda.to_gpu(self.y))

    def check_backward(self, p_data, x_data, y_data, grad):
        gradient_check.check_backward(
            functions.LinearInterpolate(), (p_data, x_data, y_data), grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.p, self.x, self.y, self.g)

    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.p),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.y),
                            cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

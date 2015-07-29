import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class NumeraicalGradientTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 1)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 1)).astype(numpy.float32)
        self.f = lambda x: (x**2, )
        self.df = lambda x: (2*x, )

    def check_numerical_grad(self, f, df, x, dy):
        dx_expect, = tuple(d*dy for d in df(x))
        func = lambda: f(x)
        dx_actual, = gradient_check.numerical_grad(func, (x,), (dy,), eps=0.5)
        gradient_check.assert_allclose(dx_expect, dx_actual)

    @condition.retry(3)
    def test_numerical_grad_cpu(self):
        self.check_numerical_grad(self.f, self.df, self.x, self.gy)

    @condition.retry(3)
    @attr.gpu
    def test_numerical_grad_gpu(self):
        self.check_numerical_grad(self.f, self.df, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


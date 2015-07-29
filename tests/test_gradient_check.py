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

    def _f(self, x):
        return x**2

    def _df(self, x):
        return 2*x

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 1)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 1)).astype(numpy.float32)
        self.f = lambda x: (self._f(x),)
        self.df = lambda x: (self._df(x),)

    def check_numerical_grad(self, f, df, x, dy):
        dx_expect, = tuple(d*dy for d in df(x))
        func = lambda: f(x)
        dx_actual, = gradient_check.numerical_grad(func, (x,), (dy,))
        gradient_check.assert_allclose(dx_expect, dx_actual)

    @condition.retry(3)
    def test_numerical_grad_cpu(self):
        self.check_numerical_grad(self.f, self.df, self.x, self.gy)

    @condition.retry(3)
    @attr.gpu
    def test_numerical_grad_gpu(self):
        self.check_numerical_grad(self.f, self.df, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class NumericalGradientTest2(NumeraicalGradientTest):

    _f = lambda self, x: 1
    _df = lambda self, x: 0


class NumericalGradientTest3(NumeraicalGradientTest):

    def _exp(self, x):
        if isinstance(x, numpy.ndarray):
            return numpy.exp(x)
        else:
            return cuda.cumath.exp(x)

    _f = _df = _exp


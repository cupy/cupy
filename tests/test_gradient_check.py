import unittest

import numpy

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
        self.check_numerical_grad(self.f, self.df,
                                  cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


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


class AssertAllCloseTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_identical(self, x):
        gradient_check.assert_allclose(x, x, atol=0, rtol=0)

    @condition.repeat(5)
    def test_identical_cpu(self):
        self.check_identical(self.x)

    @condition.repeat(5)
    @attr.gpu
    def test_identical_gpu(self):
        self.check_identical(cuda.to_gpu(self.x))

    def check_atol(self, x, y):
        x_cpu = cuda.to_cpu(x)
        y_cpu = cuda.to_cpu(y)
        max_abs_diff = numpy.max(numpy.abs(x_cpu-y_cpu))
        with self.assertRaises(AssertionError):
            gradient_check.assert_allclose(x, y, atol=max_abs_diff - 1, rtol=0)
        gradient_check.assert_allclose(x, y, atol=max_abs_diff + 1, rtol=0)

    @condition.repeat(5)
    def test_atol_cpu(self):
        self.check_atol(self.x, self.y)

    @condition.repeat(5)
    @attr.gpu
    def test_atol_gpu(self):
        self.check_atol(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


class AssertAllCloseTest2(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.y = numpy.random.uniform(1, 2, (2, 3)).astype(numpy.float32)

    def check_rtol(self, x, y):
        x_cpu = cuda.to_cpu(x)
        y_cpu = cuda.to_cpu(y)
        max_ratio = numpy.max(numpy.abs(x_cpu-y_cpu)/y_cpu)
        with self.assertRaises(AssertionError):
            gradient_check.assert_allclose(x, y, atol=0, rtol=max_ratio - 1)
        gradient_check.assert_allclose(x, y, atol=0, rtol=max_ratio + 1)

    @condition.repeat(5)
    def test_rtol_cpu(self):
        self.check_rtol(self.x, self.y)

    @condition.repeat(5)
    @attr.gpu
    def test_rtol_gpu(self):
        self.check_rtol(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

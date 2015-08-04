import unittest

import numpy
import six

from chainer import cuda
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

class NumeraicalGradientTest(unittest.TestCase):

    def f(self, xs):
        return (xs[0] ** 2,)

    def df(self, xs):
        return ((2 * xs[0],),)

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (_uniform(2, 1),)

    def check_numerical_grad(self, f, df, xs, gys):
        dfxs = df(xs)

        gys = tuple(0 if gy is None else gy for gy in gys)
        # matrix-vector multiplication of dfxs and dys
        dx_expect = map(lambda dfx: sum(map(lambda (a, b): a*b, zip(dfx, gys))), dfxs)

        func = lambda: f(xs)
        dx_actual = gradient_check.numerical_grad(func, xs, gys)

        self.assertEqual(len(dx_expect), len(dx_actual))
        for e, a in zip(dx_expect, dx_actual):
            gradient_check.assert_allclose(e, a)

    @condition.retry(3)
    def test_numerical_grad_cpu(self):
        self.check_numerical_grad(self.f, self.df, self.xs, self.gys)

    @condition.retry(3)
    @attr.gpu
    def test_numerical_grad_gpu(self):
        gys = tuple(None if gy is None else cuda.to_gpu(gy) for gy in self.gys)

        self.check_numerical_grad(self.f, self.df,
                                  map(cuda.to_gpu, self.xs), gys)


class NumericalGradientTest2(NumeraicalGradientTest):

    f = lambda self, xs: (1,)
    df = lambda self, xs: ((0,),)


def _exp(x):
    if isinstance(x, numpy.ndarray):
        return numpy.exp(x)
    else:
        return cuda.cumath.exp(x)


class NumericalGradientTest3(NumeraicalGradientTest):

    def f(self, xs):
        return (_exp(xs[0]),)

    def df(self, xs):
        return ((_exp(xs[0]),),)

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (_uniform(2, 1),)


def _full_like(x, val):
    if isinstance(x, numpy.ndarray):
        return numpy.full_like(x, val)
    else:
        return cuda.full_like(x, val)


class NumericalGradientTest4(NumeraicalGradientTest):

    def f(self, xs):
        assert len(xs) == 2
        return (2 * xs[0] + 3 * xs[1], 4 * xs[0] + 5 * xs[1], 6 * xs[0] + 7 * xs[1])


    def df(self, xs):
        assert len(xs) == 2
        return ((_full_like(xs[0], 2), _full_like(xs[0], 4), _full_like(xs[0], 6)),
                (_full_like(xs[1], 3), _full_like(xs[1], 5), _full_like(xs[1], 7)))

    def setUp(self):
        self.xs = tuple(_uniform(2, 1) for _ in six.moves.range(2))
        self.gys = tuple(_uniform(2, 1) for _ in six.moves.range(3))


class NumericalGradientTest5(NumeraicalGradientTest):

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (None,)


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

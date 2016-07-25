import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.normalization import batch_normalization
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _batch_normalization(expander, gamma, beta, x, mean, var):
    mean = mean[expander]
    std = numpy.sqrt(var)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return y_expect


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
        self.eps = 1e-5

        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        self.args = [self.x, self.gamma, self.beta]
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_optionss = {
            'eps': 1e-2, 'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_optionss = {
                'eps': 2 ** -3, 'atol': 5e-2, 'rtol': 1e-1}

    def check_forward(self, args):
        y = functions.batch_normalization(
            *[chainer.Variable(i) for i in args], eps=self.eps)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_optionss)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    def check_backward(self, args, y_grad):
        gradient_check.check_backward(
            batch_normalization.BatchNormalizationFunction(eps=self.eps),
            args, y_grad, **self.check_backward_optionss)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestFixedBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.expander = (None, Ellipsis) + (None,) * self.ndim

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.eps = 1e-5
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.var = numpy.random.uniform(
            0.5, 1, (3,)).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta, self.mean, self.var]
        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_optionss = {
            'eps': 1e-2, 'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_optionss = {
                'eps': 2 ** -5, 'atol': 1e-2, 'rtol': 1e-1}

    def check_forward(self, args):
        y = functions.fixed_batch_normalization(
            *[chainer.Variable(i) for i in args], eps=self.eps)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_optionss)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    def check_backward(self, args, y_grad):
        gradient_check.check_backward(
            batch_normalization.BatchNormalizationFunction(eps=self.eps),
            args, y_grad, **self.check_backward_optionss)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

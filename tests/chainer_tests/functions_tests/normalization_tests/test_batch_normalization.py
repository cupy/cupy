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
}))
class TestBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
        self.eps = 2e-5
        self.decay = 0.9

        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(numpy.float32)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        self.args = [self.x, self.gamma, self.beta]
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        self.train = True

    def check_forward(self, args):
        y = batch_normalization.batch_normalization(
            *[chainer.Variable(i) for i in args], running_mean=self.mean, running_var=self.var, decay=self.decay, eps=self.eps)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        gradient_check.assert_allclose(y_expect, y.data, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    def check_backward(self, args, y_grad):
        gradient_check.check_backward(
            batch_normalization.BatchNormalizationFunction(mean=self.mean,
                                                                      var=self.var, train=self.train, decay=self.decay, eps=self.eps),
            args, y_grad, eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
}))
class TestFixedBatchNormalization(unittest.TestCase):

    def setUp(self):
        #numpy.random.seed(0) # For debugging
        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(numpy.float32)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.expander = (None, Ellipsis) + (None,) * self.ndim

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.eps = 2e-5
        self.decay = 0.0
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.var = numpy.random.uniform(
            0.5, 1, (3,)).astype(numpy.float32)
        self.args = [self.x, self.gamma, self.beta]
        self.train = False

    def check_forward(self, args):
        y = batch_normalization.fixed_batch_normalization(
            *[chainer.Variable(i) for i in args], fixed_mean=self.mean, fixed_var=self.var, eps=self.eps)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)
        gradient_check.assert_allclose(y_expect, y.data, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        print('In: test_forward_gpu()...')
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    def check_backward(self, args, y_grad):
        gradient_check.check_backward(
            batch_normalization.BatchNormalizationFunction(mean=self.mean,
                                                                      var=self.var, train=self.train, decay=self.decay, eps=self.eps),
            args, y_grad, eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

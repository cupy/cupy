import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _batch_normalization(expander, gamma, beta, x, mean, var, eps, test):
    mean = mean[expander]
    if test:
        std = numpy.sqrt(var[expander])
    else:
        std = numpy.sqrt(var[expander] + eps)
    y_expect = gamma * (x - mean) / std + beta
    return y_expect


@testing.parameterize(*testing.product({
    'test': [True, False],
    'volatile': ['on', 'off'],
    'ndim': [0, 1, 2, 3],
}))
class BatchNormalizationTest(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * (self.ndim)
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.link = links.BatchNormalization(3)
        gamma = self.link.gamma.data
        gamma[...] = numpy.random.uniform(.5, 1, gamma.shape)
        beta = self.link.beta.data
        beta[...] = numpy.random.uniform(-1, 1, beta.shape)
        self.link.zerograds()

        self.gamma = gamma.copy()[self.expander]  # fixed on CPU
        self.beta = beta.copy()[self.expander]   # fixed on CPU

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        if self.test:
            self.mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
            self.var = numpy.random.uniform(0.5, 1, (3,)).astype(numpy.float32)
            self.link.avg_mean[...] = self.mean
            self.link.avg_var[...] = self.var
        else:
            self.mean = self.x.mean(axis=self.aggr_axes)
            self.var = self.x.var(axis=self.aggr_axes)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data, volatile=self.volatile)
        y = self.link(x, test=self.test)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean,
            self.var, self.link.eps, self.test)

        gradient_check.assert_allclose(y_expect, y.data, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.gamma, self.link.beta),
            eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

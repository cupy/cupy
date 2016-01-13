import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class BatchNormalizationTestBase(object):

    def check_forward(self, x_data):
        x = chainer.Variable(x_data, volatile=self.volatile)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)

        mean = self.x.mean(axis=self.aggr_axes, keepdims=True)
        std = numpy.sqrt(
            self.x.var(axis=self.aggr_axes, keepdims=True) + self.link.eps)
        y_expect = self.gamma * (self.x - mean) / std + self.beta

        gradient_check.assert_allclose(y_expect, y.data)
        self.assertEqual(numpy.float32, y.data.dtype)

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


# fully-connected usage
@testing.parameterize(
    {'volatile': 'on'},
    {'volatile': 'off'},
)
class TestBatchNormalization(BatchNormalizationTestBase, unittest.TestCase):
    aggr_axes = 0

    def setUp(self):
        self.link = links.BatchNormalization(3)
        gamma = self.link.gamma.data
        gamma[...] = numpy.random.uniform(.5, 1, gamma.shape)
        beta = self.link.beta.data
        beta[...] = numpy.random.uniform(-1, 1, beta.shape)
        self.link.zerograds()

        self.gamma = gamma.copy().reshape(1, 3)  # fixed on CPU
        self.beta = beta.copy().reshape(1, 3)   # fixed on CPU

        self.x = numpy.random.uniform(-1, 1, (7, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (7, 3)).astype(numpy.float32)


# convolutional usage
@testing.parameterize(
    {'volatile': 'on'},
    {'volatile': 'off'},
)
class TestBatchNormalization2D(BatchNormalizationTestBase, unittest.TestCase):
    aggr_axes = 0, 2, 3

    def setUp(self):
        self.link = links.BatchNormalization(3)
        gamma = self.link.gamma.data
        gamma[...] = numpy.random.uniform(.5, 1, gamma.shape)
        beta = self.link.beta.data
        beta[...] = numpy.random.uniform(-1, 1, beta.shape)
        self.link.zerograds()

        self.gamma = gamma.copy().reshape(1, 3, 1, 1)  # fixed on CPU
        self.beta = beta.copy().reshape(1, 3, 1, 1)   # fixed on CPU

        self.x = numpy.random.uniform(-1, 1,
                                      (7, 3, 2, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (7, 3, 2, 2)).astype(numpy.float32)


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestBatchNormalization(unittest.TestCase):
    # fully-connected usage

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

        self.test = False

    def set_avg(self):
        mean = self.x.mean(axis=self.aggr_axes)
        var = self.x.var(axis=self.aggr_axes)
        if cuda.get_array_module(self.link.avg_mean) != numpy:
            mean = cuda.to_gpu(mean)
            var = cuda.to_gpu(var)
        self.link.avg_mean[...] = mean
        self.link.avg_var[...] = var

    def check_forward(self, x_data):
        if self.test:
            self.set_avg()
        x = chainer.Variable(x_data)
        y = self.link(x, test=self.test)
        self.assertEqual(y.data.dtype, numpy.float32)

        mean = self.x.mean(axis=self.aggr_axes, keepdims=True)
        std = numpy.sqrt(
            self.x.var(axis=self.aggr_axes, keepdims=True) + self.link.eps)
        y_expect = self.gamma * (self.x - mean) / std + self.beta

        gradient_check.assert_allclose(y_expect, y.data, rtol=1e-3, atol=1e-4)
        self.assertEqual(numpy.float32, y.data.dtype)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_forward_cpu_fixed(self):
        self.test = True
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_fixed(self):
        self.test = True
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        if self.test:
            self.set_avg()
        x = chainer.Variable(x_data)
        y = self.link(x, test=self.test)
        y.grad = y_grad
        y.backward()

        f = lambda: (self.link(x).data,)
        gx, ggamma, gbeta = gradient_check.numerical_grad(
            f, (x.data, self.link.gamma.data, self.link.beta.data),
            (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad, rtol=1e-3, atol=1e-4)
        gradient_check.assert_allclose(ggamma, self.link.gamma.grad)
        gradient_check.assert_allclose(gbeta, self.link.beta.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @condition.retry(3)
    def test_backward_cpu_fixed(self):
        self.test = True
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_fixed(self):
        self.test = True
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


# convolutional usage
class TestBatchNormalization2D(TestBatchNormalization):
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

        self.test = False


testing.run_module(__name__, __file__)

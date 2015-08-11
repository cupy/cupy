import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestInception(unittest.TestCase):

    in_channels = 3
    out1, proj3, out3, proj5, out5, proj_pool = 3, 2, 3, 2, 3, 3

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, self.in_channels, 5, 5)).astype(numpy.float32)
        out = self.out1 + self.out3 + self.out5 + self.proj_pool
        self.gy = numpy.random.uniform(-1, 1, (10, out, 5, 5)).astype(numpy.float32)
        self.f = functions.Inception(self.in_channels, self.out1, self.proj3, self.out3, self.proj5, self.out5, self.proj_pool)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.f(x)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @condition.retry(3)
    @attr.gpu
    def test_forward_gpu(self):
        self.f.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.f(x)
        y.grad = y_grad
        y.backward()

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @condition.retry(3)
    @attr.gpu
    def test_backward_gpu(self):
        self.f.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


def _zeros(*shape):
    return chainer.Variable(numpy.zeros(shape).astype(numpy.float32))

class TestInception2(unittest.TestCase):

    in_channels = 3
    out1, proj3, out3, proj5, out5, proj_pool = 3, 2, 3, 2, 3, 3
    batchsize = 10
    insize = 10

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, self.in_channels, 5, 5)).astype(numpy.float32)
        out = self.out1 + self.out3 + self.out5 + self.proj_pool
        self.f = functions.Inception(self.in_channels, self.out1, self.proj3, self.out3, self.proj5, self.out5, self.proj_pool)
        self.f.f.conv1 = mock.MagicMock(return_value=_zeros(self.batchsize, self.out1, self.insize, self.insize))
        self.f.f.proj3 = mock.MagicMock(return_value=_zeros(self.batchsize, self.proj3, self.insize, self.insize))
        self.f.f.conv3 = mock.MagicMock(return_value=_zeros(self.batchsize, self.out3, self.insize, self.insize))
        self.f.f.proj5 = mock.MagicMock(return_value=_zeros(self.batchsize, self.proj5, self.insize, self.insize))
        self.f.f.conv5 = mock.MagicMock(return_value=_zeros(self.batchsize, self.out5, self.insize, self.insize))
        self.f.f.projp = mock.MagicMock(return_value=_zeros(self.batchsize, self.proj_pool, self.insize, self.insize))

    def test_cpu(self):
        x = chainer.Variable(self.x)
        y = self.f(x)
        conv3_in = functions.relu(self.f.f.proj3(x))
        conv5_in = functions.relu(self.f.f.proj5(x))
        projp_in = functions.max_pooling_2d(x, 3, stride=1, pad=1)

        self.f.f.conv1.assert_called_with(x)
        self.f.f.proj3.assert_called_with(x)
        self.f.f.conv3.assert_called_with(conv3_in)
        self.f.f.proj5.assert_called_with(x)
        self.f.f.conv5.assert_called_with(conv5_in)
        self.f.f.projp.assert_called_with(projp_in)


testing.run_module(__name__, __file__)

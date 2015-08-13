import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestInceptionBackward(unittest.TestCase):

    in_channels = 3
    out1, proj3, out3, proj5, out5, proj_pool = 3, 2, 3, 2, 3, 3

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)
        out = self.out1 + self.out3 + self.out5 + self.proj_pool
        self.gy = numpy.random.uniform(
            -1, 1, (10, out, 5, 5)).astype(numpy.float32)
        self.f = functions.Inception(
            self.in_channels, self.out1, self.proj3, self.out3,
            self.proj5, self.out5, self.proj_pool)

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.f(x)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.f.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


def _ones(gpu, *shape):
    if gpu:
        return chainer.Variable(cuda.ones(shape).astype(numpy.float32))
    return chainer.Variable(numpy.ones(shape).astype(numpy.float32))


class TestInceptionForward(unittest.TestCase):

    in_channels = 3
    out1, proj3, out3, proj5, out5, proj_pool = 3, 2, 3, 2, 3, 3
    batchsize = 10
    insize = 10

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)
        self.f = functions.Inception(
            self.in_channels, self.out1,
            self.proj3, self.out3,
            self.proj5, self.out5, self.proj_pool)

    def setup_mock(self, gpu):
        self.f.f = mock.MagicMock()
        self.f.f.conv1.return_value = _ones(gpu,
                                            self.batchsize, self.out1,
                                            self.insize, self.insize)
        self.f.f.proj3.return_value = _ones(gpu,
                                            self.batchsize, self.proj3,
                                            self.insize, self.insize)
        self.f.f.conv3.return_value = _ones(gpu,
                                            self.batchsize, self.out3,
                                            self.insize, self.insize)
        self.f.f.proj5.return_value = _ones(gpu,
                                            self.batchsize, self.proj5,
                                            self.insize, self.insize)
        self.f.f.conv5.return_value = _ones(gpu, self.batchsize, self.out5,
                                            self.insize, self.insize)
        self.f.f.projp.return_value = _ones(gpu, self.batchsize,
                                            self.proj_pool, self.insize,
                                            self.insize)

    def check_call(self, x, f, gpu):
        self.setup_mock(gpu)
        f(chainer.Variable(x))

        # Variable.__eq__ raises NotImplementedError,
        # so we cannot check arguments
        expected = [mock.call.conv1(mock.ANY), mock.call.proj3(mock.ANY),
                    mock.call.conv3(mock.ANY), mock.call.proj5(mock.ANY),
                    mock.call.conv5(mock.ANY), mock.call.projp(mock.ANY)]

        self.assertListEqual(self.f.f.mock_calls, expected)

    def test_call_cpu(self):
        self.check_call(self.x, self.f, False)

    @attr.gpu
    def test_call_gpu(self):
        x = cuda.to_gpu(self.x)
        self.f.to_gpu()
        self.check_call(x, self.f, True)


testing.run_module(__name__, __file__)

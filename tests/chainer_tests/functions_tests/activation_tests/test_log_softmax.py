import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestLogSoftmax(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.log_softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.empty_like(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            x = self.x[i]
            log_z = numpy.ufunc.reduce(numpy.logaddexp, x)
            y_expect[i] = x - log_z

        gradient_check.assert_allclose(y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.log_softmax(x, use_cudnn)
        y.grad = gy_data
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad, atol=1e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


class TestLogSoftmaxUnstable(TestLogSoftmax):

    def setUp(self):
        self.x = numpy.array([[-1000, 1]], dtype=numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (1, 2)).astype(numpy.float32)


class TestReplicatedLogSoftmax1(TestLogSoftmax):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.log_softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.exp(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            for k in six.moves.range(y_expect.shape[2]):
                y_expect[i, :, k] /= y_expect[i, :, k].sum()

        y_expect = numpy.log(y_expect)
        gradient_check.assert_allclose(y_expect, y.data)


class TestReplicatedLogSoftmax2(TestLogSoftmax):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (2, 3, 4, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (2, 3, 4, 5)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.log_softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.exp(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            for k in six.moves.range(y_expect.shape[2]):
                for l in six.moves.range(y_expect.shape[3]):
                    y_expect[i, :, k, l] /= y_expect[i, :, k, l].sum()

        y_expect = numpy.log(y_expect)
        gradient_check.assert_allclose(y_expect, y.data)


testing.run_module(__name__, __file__)

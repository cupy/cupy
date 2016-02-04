import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.empty_like(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            x = self.x[i]
            log_z = numpy.ufunc.reduce(numpy.logaddexp, x)
            x -= log_z
            y_expect[i] = numpy.exp(x)

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
        gradient_check.check_backward(
            functions.Softmax(use_cudnn), x_data, gy_data, eps=1e-2)

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


class TestSoftmaxUnstable(TestSoftmax):

    def setUp(self):
        self.x = numpy.array([[-1000, 1]], dtype=numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (1, 2)).astype(numpy.float32)


class TestReplicatedSoftmax1(TestSoftmax):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.exp(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            for k in six.moves.range(y_expect.shape[2]):
                y_expect[i, :, k] /= y_expect[i, :, k].sum()

        gradient_check.assert_allclose(y_expect, y.data)


class TestReplicatedSoftmax2(TestSoftmax):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (2, 3, 4, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (2, 3, 4, 5)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.exp(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            for k in six.moves.range(y_expect.shape[2]):
                for l in six.moves.range(y_expect.shape[3]):
                    y_expect[i, :, k, l] /= y_expect[i, :, k, l].sum()

        gradient_check.assert_allclose(y_expect, y.data)


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
@attr.cudnn
class TestSoftmaxCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.softmax(x, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.softmaxForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    def test_call_cudnn_backrward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.softmaxBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn)


testing.run_module(__name__, __file__)

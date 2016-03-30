import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(
    {'x': numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)},
    {'x': numpy.array([[-1000, 1]], dtype=numpy.float32)},
    {'x': numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)},
    {'x': numpy.random.uniform(-1, 1, (2, 3, 4, 5)).astype(numpy.float32)},
)
class TestLogSoftmax(unittest.TestCase):

    def setUp(self):
        self.gy = numpy.random.uniform(
            -1, 1, self.x.shape).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.log_softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)

        log_z = numpy.ufunc.reduce(
            numpy.logaddexp, self.x, axis=1, keepdims=True)
        y_expect = self.x - log_z

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
            functions.LogSoftmax(use_cudnn), x_data, gy_data,
            eps=1e-2, atol=1e-3)

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


testing.run_module(__name__, __file__)

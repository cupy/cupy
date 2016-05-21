import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [None, (2, 3), (2, 2, 3), (2, 2, 2, 3)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLogSoftmax(unittest.TestCase):

    def setUp(self):
        if self.shape is None:
            value = -5 if self.dtype == numpy.float16 else -1000
            self.x = numpy.array([[value, 1]], dtype=self.dtype)
        else:
            self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.x.shape).astype(self.dtype)

        self.check_forward_option = {}
        self.check_backward_option = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_option = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_option = {'atol': 5e-2, 'rtol': 1e-1}

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.log_softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)

        log_z = numpy.ufunc.reduce(
            numpy.logaddexp, self.x, axis=1, keepdims=True)
        y_expect = self.x - log_z

        gradient_check.assert_allclose(
            y_expect, y.data, **self.check_forward_option)

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
            eps=1e-2, **self.check_backward_option)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


testing.run_module(__name__, __file__)

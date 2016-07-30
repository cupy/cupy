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


@testing.parameterize(*testing.product({
    'shape': [None, (2, 3), (2, 3, 4), (2, 3, 4, 5)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSoftmax(unittest.TestCase):

    def setUp(self):
        if self.shape is None:
            # For checking numerical stability
            value = -5 if self.dtype == numpy.float16 else -1000
            self.x = numpy.array([[value, 1]], dtype=self.dtype)
        else:
            self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.x.shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.softmax(x, use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = numpy.exp(self.x)
        y_roll = numpy.rollaxis(y_expect, 1, y_expect.ndim)
        for i in numpy.ndindex(y_roll.shape[:-1]):
            y_roll[i] /= y_roll[i].sum()

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        gradient_check.check_backward(
            functions.Softmax(use_cudnn), x_data, gy_data,
            **self.check_backward_options)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestSoftmaxCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.expect = self.use_cudnn and (
            cuda.cudnn.cudnn.getVersion() >= 3000 or
            self.dtype != numpy.float16)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.softmax(x, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.softmaxForward') as func:
            self.forward()
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.softmaxBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)

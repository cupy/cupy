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


class TestTanh(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-.5, .5, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.tanh(x, use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = functions.tanh(chainer.Variable(self.x))
        gradient_check.assert_allclose(y_expect.data, y.data)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), True)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)), True)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        gradient_check.check_backward(
            functions.Tanh(use_cudnn), x_data, gy_data)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
                            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
@attr.cudnn
class TestTanhCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.tanh(x, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.activationForward_v3') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    def test_call_cudnn_backrward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.activationBackward_v3') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn)


testing.run_module(__name__, __file__)

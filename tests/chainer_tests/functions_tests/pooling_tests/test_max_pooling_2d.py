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


@testing.parameterize(*testing.product({
    'cover_all': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestMaxPooling2D(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical gradient
        self.x = numpy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        if self.cover_all:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 3, 3, 2)).astype(self.dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 3, 2, 2)).astype(self.dtype)
        self.check_backward_options = {'eps': 2.0 ** -8}

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                     cover_all=self.cover_all,
                                     use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                if self.cover_all:
                    expect = numpy.array([
                        [x[0:2, 0:2].max(), x[0:2, 1:3].max()],
                        [x[1:4, 0:2].max(), x[1:4, 1:3].max()],
                        [x[3:4, 0:2].max(), x[3:4, 1:3].max()]])
                else:
                    expect = numpy.array([
                        [x[0:2, 0:2].max(), x[0:2, 1:3].max()],
                        [x[1:4, 0:2].max(), x[1:4, 1:3].max()]])
                testing.assert_allclose(expect, y_data[k, c])

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu_wide(self):  # see #120
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.MaxPooling2D(
                3, stride=2, pad=1, cover_all=self.cover_all,
                use_cudnn=use_cudnn),
            x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)

    def test_backward_cpu_more_than_once(self):
        func = functions.MaxPooling2D(
            3, stride=2, pad=1, cover_all=self.cover_all)
        func(self.x)
        func.backward_cpu((self.x,), (self.gy,))
        func.backward_cpu((self.x,), (self.gy,))


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestMaxPooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1,
                                           (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False, use_cudnn=self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports max-pooling2d')
    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.poolingForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports max-pooling2d')
    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn)


testing.run_module(__name__, __file__)

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
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSpatialPyramidPooling2D(unittest.TestCase):
    pyramid_height = 3
    output_dim = 63  # channels(c=3) * (1 + 4 + 16) = 63
    n, c, h, w = 2, 3, 9, 8
    pooling_class = functions.MaxPooling2D

    def setUp(self):
        # Spacial pyramid pooling uses max pooling in its implementation.
        # To avoid instability of numerical gradient, use enough different
        # values.
        shape = (self.n, self.c, self.h, self.w)
        size = numpy.prod(shape)
        self.x = numpy.arange(size, dtype=self.dtype).reshape(shape)
        numpy.random.shuffle(self.x)
        self.x += numpy.random.uniform(
            0.4, 0.6, shape).astype(self.dtype)
        self.x /= size

        self.one = numpy.ones(
            (self.n, self.c, self.h, self.w)).astype(self.dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (self.n, self.output_dim, 1, 1)).astype(self.dtype)
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.spatial_pyramid_pooling_2d(
            x, self.pyramid_height, self.pooling_class,
            use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    def check_forward_ones(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.spatial_pyramid_pooling_2d(
            x, self.pyramid_height, self.pooling_class, use_cudnn=use_cudnn)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(y_data.shape, (self.n, self.output_dim, 1, 1))
        self.assertEqual(y_data.dtype, self.dtype)
        testing.assert_allclose(y_data, numpy.ones_like(y_data))

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)
        self.check_forward_ones(self.one)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))
        self.check_forward_ones(cuda.to_gpu(self.one))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)
        self.check_forward_ones(cuda.to_gpu(self.one), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.SpatialPyramidPooling2D(
                x_data.shape[1:], self.pyramid_height, self.pooling_class,
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


class TestInvalidDtype(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.randn(5, 3, 5, 5)
        self.v = chainer.Variable(self.x.astype(numpy.int32))

    def check_invalid_dtype(self):
        functions.spatial_pyramid_pooling_2d(
            self.v, 3, functions.MaxPooling2D)

    def test_invalid_dtype_cpu(self):
        with self.assertRaises(type_check.InvalidType):
            self.check_invalid_dtype()

    @attr.gpu
    def test_invalid_dtype_gpu(self):
        self.v.to_gpu()
        with self.assertRaises(type_check.InvalidType):
            self.check_invalid_dtype()


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestMaxPooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        shape = (2, 3, 9, 8)
        size = 2 * 3 * 9 * 8
        self.x = cuda.cupy.arange(size, dtype=self.dtype).reshape(shape)
        self.gy = cuda.cupy.random.uniform(
            -1, 1, (2, 63, 1, 1)).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.spatial_pyramid_pooling_2d(
            x, 3, functions.MaxPooling2D,
            use_cudnn=self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports spatial-pyramid-pooling2d')
    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.poolingForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports spatial-pyramid-pooling2d')
    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn)


testing.run_module(__name__, __file__)

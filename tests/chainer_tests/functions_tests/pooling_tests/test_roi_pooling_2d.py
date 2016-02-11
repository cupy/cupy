import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestROIPooling2D(unittest.TestCase):

    def setUp(self):
        N = 4
        n_channels = 3
        self.x = numpy.random.randn(N, n_channels, 12, 8).astype(numpy.float32)
        self.rois = numpy.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ], dtype=numpy.float32)
        self.outh, self.outw = 5, 7
        self.spatial_scale = 0.6
        self.gy = numpy.random.uniform(
            -1, 1, (N, n_channels, self.outh, self.outw)).astype(numpy.float32)

    def check_forward(self, x_data, roi_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = functions.roi_pooling_2d(
            x, rois, outh=self.outh, outw=self.outw,
            spatial_scale=self.spatial_scale)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois))

    def check_backward(self, x_data, roi_data, y_grad):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = functions.roi_pooling_2d(x, rois, outh=self.outh, outw=self.outw,
                                     spatial_scale=self.spatial_scale)
        y.grad = y_grad
        y.backward()

        xs = (x.data, rois.data)

        def f():
            func = y.creator
            return func.forward(xs)

        gx, _ = gradient_check.numerical_grad(f, xs, (y.grad,))
        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

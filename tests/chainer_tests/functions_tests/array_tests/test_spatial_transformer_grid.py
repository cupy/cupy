import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
class TestSpatialTransformerGrid(unittest.TestCase):

    def setUp(self):
        B = 3
        self.theta = numpy.random.uniform(size=(B, 2, 3)).astype(numpy.float32)
        self.output_shape = (5, 6)
        self.grads = numpy.random.uniform(
            size=(B, 2) + self.output_shape).astype(self.theta.dtype)

    def check_forward(self, theta, output_shape, use_cudnn=True):
        grid = functions.spatial_transformer_grid(
            theta, output_shape, use_cudnn).data

        theta = cuda.to_cpu(theta)
        B = theta.shape[0]
        H, W = output_shape

        expected = []
        for b in range(B):
            for i in numpy.linspace(-1., 1., H):
                for j in numpy.linspace(-1., 1., W):
                    coord = numpy.array([j, i, 1])
                    expected.append(self.theta[b].dot(coord))
        expected = numpy.array(
            expected).reshape(B, H, W, 2).transpose(0, 3, 1, 2)
        testing.assert_allclose(grid, expected)
        self.assertEqual(grid.dtype, theta.dtype)

    def test_forward_cpu(self):
        self.check_forward(self.theta, self.output_shape)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.theta), self.output_shape, self.use_cudnn)

    def check_backward(self, theta, output_shape, grads, use_cudnn=True):
        gradient_check.check_backward(
            functions.SpatialTransformerGrid(output_shape, use_cudnn),
            (theta,), (grads,))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.theta, self.output_shape, self.grads)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.theta),
                            self.output_shape,
                            cuda.to_gpu(self.grads),
                            self.use_cudnn)


testing.run_module(__name__, __file__)

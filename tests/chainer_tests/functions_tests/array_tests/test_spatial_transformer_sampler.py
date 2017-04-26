import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _identiy_grid(in_shape):
    mesh = numpy.meshgrid(
        numpy.linspace(-1., 1., num=in_shape[2]),
        numpy.linspace(-1., 1., num=in_shape[3]))
    grid = numpy.concatenate([mesh[0][None], mesh[1][None]], axis=0)
    grid = numpy.repeat(grid[None], in_shape[0], axis=0).astype(numpy.float32)
    return grid


def _rotate_grid(in_shape):
    mesh = numpy.meshgrid(
        numpy.linspace(-1., 1., num=in_shape[2]),
        numpy.linspace(-1., 1., num=in_shape[3]))
    mesh = [numpy.rot90(mesh[0]), numpy.rot90(mesh[1])]
    grid = numpy.concatenate([mesh[0][None], mesh[1][None]], axis=0)
    grid = numpy.repeat(grid[None], in_shape[0], axis=0).astype(numpy.float32)
    return grid


def _rotate_BCHW(x):
    rotated_xs = []
    for i in range(x.shape[0]):
        x_i = x[i].transpose(1, 2, 0)
        x_i = numpy.rot90(x_i)
        rotated_xs.append(x_i.transpose(2, 0, 1))
    rotated_xs = numpy.concatenate([r_x[None] for r_x in rotated_xs], axis=0)
    return rotated_xs


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
class TestSpatialTransformerSampler(unittest.TestCase):

    in_shape = (2, 2, 4, 4)
    out_shape = (2, 2, 3, 3)
    grid_shape = (2, 2, 3, 3)

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(numpy.float32)
        self.grid = numpy.random.uniform(
            low=-1.2, high=1.2, size=self.grid_shape).astype(numpy.float32)
        self.grads = numpy.random.uniform(
            size=self.out_shape).astype(numpy.float32)

    def check_forward(self, x, grid, use_cudnn=True):
        y = functions.spatial_transformer_sampler(x, grid, use_cudnn)
        self.assertEqual(y.shape, self.out_shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.grid)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid),
                           self.use_cudnn)

    def check_backward(self, x, grid, grads, use_cudnn=True):
        gradient_check.check_backward(
            functions.SpatialTransformerSampler(use_cudnn),
            (x, grid), (grads,), atol=1e-2, rtol=1e-3, eps=1e-5)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.grid, self.grads)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.grid),
                            cuda.to_gpu(self.grads),
                            self.use_cudnn)


@testing.parameterize(
    {'grid_creator': _identiy_grid, 'operator': lambda x: x,
     'use_cudnn': True},
    {'grid_creator': _identiy_grid, 'operator': lambda x: x,
     'use_cudnn': False},
    {'grid_creator': _rotate_grid, 'operator': _rotate_BCHW,
     'use_cudnn': True},
    {'grid_creator': _rotate_grid, 'operator': _rotate_BCHW,
     'use_cudnn': False},
)
class TestSpatialTransformerSamplerForwardToyCases(unittest.TestCase):

    in_shape = (2, 2, 4, 4)
    grid_shape = (2, 2, 3, 3)

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(numpy.float32)
        self.grid = self.grid_creator(self.in_shape)

    def check_forward(self, x, grid, use_cudnn=True):
        y = functions.spatial_transformer_sampler(x, grid, use_cudnn)
        testing.assert_allclose(y.data, self.operator(self.x))

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.grid)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid),
                           self.use_cudnn)


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
class TestSpatialTransformerSamplerForwardPaddedImage(unittest.TestCase):

    in_shape = (1, 2, 4, 4)

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(numpy.float32)
        p1 = [[-0.5], [-0.5]]
        p2 = [[3.5], [3.5]]
        p3 = [[2], [3.5]]
        p4 = [[-0.5], [2]]
        self.grid = numpy.concatenate((p1, p2, p3, p4), axis=1)  # (2, 4)
        self.grid = self.grid.reshape(1, 2, 4, 1).astype(numpy.float32)
        self.grid[:, 0] =\
            ((self.grid[:, 0] / (self.in_shape[3] - 1)) - 0.5) * 2
        self.grid[:, 1] =\
            ((self.grid[:, 1] / (self.in_shape[2] - 1)) - 0.5) * 2

        exp_p1 = self.x[0, :, 0, 0] / 4
        exp_p2 = self.x[0, :, 3, 3] / 4
        exp_p3 = self.x[0, :, 3, 2] / 2
        exp_p4 = self.x[0, :, 2, 0] / 2

        self.expected = numpy.concatenate(
            (exp_p1[:, None],
             exp_p2[:, None],
             exp_p3[:, None],
             exp_p4[:, None]), axis=1)
        self.expected = self.expected.reshape(1, 2, 4, 1).astype(numpy.float32)

    def check_forward(self, x, grid, expected, use_cudnn=True):
        y = functions.spatial_transformer_sampler(x, grid, use_cudnn)

        testing.assert_allclose(y.data, expected)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.grid, self.expected)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.grid),
                           cuda.to_gpu(self.expected),
                           self.use_cudnn)


testing.run_module(__name__, __file__)

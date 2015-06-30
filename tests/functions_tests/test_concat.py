import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class Concat(unittest.TestCase):

    def setUp(self):
        self.y0 = numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)
        self.xs0 = [self.y0[:, :2], self.y0[:, 2:5], self.y0[:, 5:]]

        self.y1 = numpy.arange(21, dtype=numpy.float32).reshape(7, 3)
        self.xs1 = [self.y1[:2], self.y1[2:5], self.y1[5:]]

    def check_forward(self, xs_data, y_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        gradient_check.assert_allclose(y_data, y.data, atol=0, rtol=0)

    def test_forward_cpu_0(self):
        self.check_forward(self.xs0, self.y0, axis=1)

    def test_forward_cpu_1(self):
        self.check_forward(self.xs1, self.y1, axis=0)

    @attr.gpu
    def test_forward_gpu_0(self):
        self.check_forward(
            [cuda.to_gpu(x.copy()) for x in self.xs0],
            cuda.to_gpu(self.y0), axis=1)

    @attr.gpu
    def test_forward_gpu_1(self):
        self.check_forward(
            [cuda.to_gpu(x.copy()) for x in self.xs1],
            cuda.to_gpu(self.y1), axis=0)

    def check_backward(self, xs_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        y.grad = y.data
        y.backward()

        for x in xs:
            gradient_check.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu_0(self):
        self.check_backward(self.xs0, axis=1)

    def test_backward_cpu_1(self):
        self.check_backward(self.xs1, axis=0)

    @attr.gpu
    def test_backward_gpu_0(self):
        self.check_backward([cuda.to_gpu(x.copy()) for x in self.xs0], axis=1)

    @attr.gpu
    def test_backward_gpu_1(self):
        self.check_backward([cuda.to_gpu(x.copy()) for x in self.xs1], axis=0)

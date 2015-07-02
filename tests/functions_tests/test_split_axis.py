import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class Split(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)
        self.ys0 = [self.x0[:, :2], self.x0[:, 2:5], self.x0[:, 5:]]
        self.ys0_section = [2, 5]

        self.x1 = numpy.arange(21, dtype=numpy.float32).reshape(7, 3)
        self.ys1 = [self.x1[:2], self.x1[2:5], self.x1[5:]]
        self.ys1_section = [2, 5]

        self.x2 = numpy.arange(54, dtype=numpy.float32).reshape(2, 9, 3)
        self.ys2 = [self.x2[:, :3], self.x2[:, 3:6], self.x2[:, 6:]]
        self.ys2_section = 3

    def check_forward(self, x_data, ys_data, indices_or_sections, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(x, indices_or_sections, axis)
        for yd, y in zip(ys_data, ys):
            gradient_check.assert_allclose(yd, y.data, atol=0, rtol=0)

    def test_forward_cpu_0(self):
        self.check_forward(self.x0, self.ys0, self.ys0_section, 1)

    def test_forward_cpu_1(self):
        self.check_forward(self.x1, self.ys1, self.ys1_section, 0)

    def test_forward_cpu_2(self):
        self.check_forward(self.x2, self.ys2, self.ys2_section, 1)

    @attr.gpu
    def test_forward_gpu_0(self):
        self.check_forward(
            cuda.to_gpu(self.x0),
            [cuda.to_gpu(y.copy()) for y in self.ys0],
            self.ys0_section, axis=1)

    @attr.gpu
    def test_forward_gpu_1(self):
        self.check_forward(
            cuda.to_gpu(self.x1),
            [cuda.to_gpu(y.copy()) for y in self.ys1],
            self.ys1_section, axis=0)

    @attr.gpu
    def test_forward_gpu_2(self):
        self.check_forward(
            cuda.to_gpu(self.x2),
            [cuda.to_gpu(y.copy()) for y in self.ys2],
            self.ys2_section, axis=1)

    def check_backward(self, x_data, indices_or_sections, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(x, indices_or_sections, axis)
        for y in ys:
            y.grad = y.data
        ys[0].backward()

        gradient_check.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu_0(self):
        self.check_backward(self.x0, self.ys0_section, axis=1)

    def test_backward_cpu_1(self):
        self.check_backward(self.x1, self.ys1_section, axis=0)

    def test_backward_cpu_2(self):
        self.check_backward(self.x2, self.ys2_section, axis=1)

    @attr.gpu
    def test_backward_gpu_0(self):
        self.check_backward(cuda.to_gpu(self.x0), self.ys0_section, axis=1)

    @attr.gpu
    def test_backward_gpu_1(self):
        self.check_backward(cuda.to_gpu(self.x1), self.ys1_section, axis=0)

    @attr.gpu
    def test_backward_gpu_2(self):
        self.check_backward(cuda.to_gpu(self.x2), self.ys2_section, axis=1)

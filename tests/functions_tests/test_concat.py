from unittest import TestCase

import numpy
from pycuda.gpuarray import to_gpu, GPUArray

from chainer import Variable
from chainer.gradient_check import assert_allclose
from chainer.functions import concat

class Concat(TestCase):
    def setUp(self):
        self.y0 = numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)
        self.xs0 = [self.y0[:, :2], self.y0[:, 2:5], self.y0[:, 5:]]

        self.y1 = numpy.arange(21, dtype=numpy.float32).reshape(7, 3)
        self.xs1 = [self.y1[:2], self.y1[2:5], self.y1[5:]]

    def check_forward(self, xs_data, y_data, axis):
        xs = tuple(Variable(x_data) for x_data in xs_data)
        y  = concat(xs, axis=axis)
        assert_allclose(y_data, y.data, atol=0, rtol=0)

    def test_forward_cpu_0(self):
        self.check_forward(self.xs0, self.y0, axis=1)

    def test_forward_cpu_1(self):
        self.check_forward(self.xs1, self.y1, axis=0)

    def test_forward_gpu_0(self):
        self.check_forward(
            [to_gpu(x.copy()) for x in self.xs0], to_gpu(self.y0), axis=1)

    def test_forward_gpu_1(self):
        self.check_forward(
            [to_gpu(x.copy()) for x in self.xs1], to_gpu(self.y1), axis=0)

    def check_backward(self, xs_data, axis):
        xs = tuple(Variable(x_data) for x_data in xs_data)
        y  = concat(xs, axis=axis)
        y.grad = y.data
        y.backward()

        for x in xs:
            assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu_0(self):
        self.check_backward(self.xs0, axis=1)

    def test_backward_cpu_1(self):
        self.check_backward(self.xs1, axis=0)

    def test_backward_gpu_0(self):
        self.check_backward([to_gpu(x.copy()) for x in self.xs0], axis=1)

    def test_backward_gpu_1(self):
        self.check_backward([to_gpu(x.copy()) for x in self.xs1], axis=0)

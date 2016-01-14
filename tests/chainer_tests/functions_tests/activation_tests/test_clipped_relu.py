import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestClippedReLU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        # Avoid values around zero and z for stability of numerical gradient
        for i in range(self.x.size):
            if -0.01 < self.x.flat[i] < 0.01 or 0.74 < self.x.flat[i] < 0.76:
                self.x.flat[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.z = 0.75

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.clipped_relu(x, self.z)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                y_expect[i] = 0
            elif self.x[i] > self.z:
                y_expect[i] = self.z

        gradient_check.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.ClippedReLU(self.z), x_data, y_grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestClippedReLUZeroDim(TestClippedReLU):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        # Avoid values around zero and z for stability of numerical gradient
        if -0.01 < self.x[()] < 0.01 or 0.74 < self.x[()] < 0.76:
            self.x[()] = 0.5
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        self.z = 0.75

testing.run_module(__name__, __file__)

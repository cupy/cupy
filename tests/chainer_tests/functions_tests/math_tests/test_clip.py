import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestClip(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        # Avoid values around x_min and x_max for stability of numerical
        # gradient
        for i, j in numpy.ndindex(self.x.shape):
            if -0.76 < self.x[i][j] < -0.74:
                self.x[i][j] = -0.5
            elif 0.74 < self.x[i][j] < 0.76:
                self.x[i][j] = 0.5
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.x_min = -0.75
        self.x_max = 0.75

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.clip(x, self.x_min, self.x_max)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < self.x_min:
                y_expect[i] = self.x_min
            elif self.x[i] > self.x_max:
                y_expect[i] = self.x_max

        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Clip(self.x_min, self.x_max), x_data, y_grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestClipZeroDim(TestClip):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        # Avoid values around x_min and x_max for stability of numerical
        # gradient
        if -0.76 < self.x[()] < -0.74:
            self.x[()] = -0.5
        elif 0.74 < self.x[()] < 0.76:
            self.x[()] = 0.5
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        self.x_min = -0.75
        self.x_max = 0.75


class TestClipInvalidInterval(unittest.TestCase):

    def test_invalid_interval(self):
        with self.assertRaises(AssertionError):
            functions.Clip(1.0, -1.0)

testing.run_module(__name__, __file__)

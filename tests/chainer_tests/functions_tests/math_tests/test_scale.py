import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestScale(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        self.axis = 1
        self.y_expected = numpy.copy(self.x1)
        for i, j, k in numpy.ndindex(self.y_expected.shape):
            self.y_expected[i, j, k] *= self.x2[j]
        self.gy = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)

    def check_forward(self, x1_data, x2_data, axis, y_expected):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.scale(x1, x2, axis)
        testing.assert_allclose(y_expected, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2, self.axis, self.y_expected)

    @attr.gpu
    def test_forward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        self.check_forward(x1, x2, self.axis, self.y_expected)

    def check_backward(self, x1_data, x2_data, axis, y_grad):
        x = (x1_data, x2_data)
        gradient_check.check_backward(
            lambda x, y: functions.scale(x, y, axis),
            x, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.axis, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x1, x2, self.axis, gy)


class TestScaleInvalidShape(unittest.TestCase):

    def test_scale_invalid_shape(self):
        x1 = chainer.Variable(numpy.zeros((3, 2, 3), numpy.float32))
        x2 = chainer.Variable(numpy.zeros((2), numpy.float32))
        axis = 0
        with chainer.DebugMode(True):
            with self.assertRaises(AssertionError):
                functions.scale(x1, x2, axis)

testing.run_module(__name__, __file__)

import unittest

import numpy
import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestMaximum(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.y_expected = numpy.maximum(self.x1, self.x2)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_forward(self, x1_data, x2_data, y_expected):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.maximum(x1, x2)
        gradient_check.assert_allclose(y_expected, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2, self.y_expected)

    @attr.gpu
    def test_forward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        self.check_forward(x1, x2, self.y_expected)

    def check_backward(self, x1_data, x2_data, y_grad):
        func = functions.maximum
        x = (x1_data, x2_data)
        gradient_check.check_backward(func, x, y_grad)

    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x1, x2, gy)


class TestMaximumZeroDim(TestMaximum):

    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        self.y_expected = numpy.maximum(self.x1, self.x2)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

testing.run_module(__name__, __file__)

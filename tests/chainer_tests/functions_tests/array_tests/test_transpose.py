import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestTranspose(unittest.TestCase):
    axes = (-1, 0, 1)

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.gy = numpy.random.uniform(-1, 1, (2, 4, 3))

    def check_forward(self, x_data):
        axes = self.axes
        x = chainer.Variable(x_data)
        y = functions.transpose(x, axes)
        self.assertEqual(y.data.dtype, numpy.float)
        self.assertTrue((self.x.transpose(axes) == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Transpose(self.axes),
            x_data, y_grad, eps=1e-5, rtol=1e-5)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestTransposeNoneAxes(TestTranspose):
    axes = None

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.gy = numpy.random.uniform(-1, 1, (2, 3, 4))


testing.run_module(__name__, __file__)

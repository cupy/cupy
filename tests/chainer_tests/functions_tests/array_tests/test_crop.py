import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestCrop(unittest.TestCase):
    axes = [1, 2]
    offsets = 0

    def setUp(self):
        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.shape = (4, 2, 1)
        self.gy_data = numpy.random.uniform(-1, 1, self.shape)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.crop(x, self.shape, self.axes, self.offsets)
        self.assertEqual(y.data.dtype, numpy.float)
        numpy.testing.assert_equal(cuda.to_cpu(x_data)[:, :2, :1],
                                   cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Crop(self.shape, self.axes, self.offsets),
            (x_data,), y_grad)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.gy_data))


testing.run_module(__name__, __file__)

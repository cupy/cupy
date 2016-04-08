import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestReshape(unittest.TestCase):
    out_shape = (2, 2, 6)

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 2, 6)).astype(numpy.float32)

    def check_forward(self, x_data):
        shape = self.out_shape
        x = chainer.Variable(x_data)
        y = functions.reshape(x, shape)
        self.assertEqual(y.data.dtype, numpy.float32)
        self.assertTrue((self.x.reshape(shape) == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Reshape(self.gy.shape), x_data, y_grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestReshapeUnknownDimension(TestReshape):
    out_shape = (2, -1, 6)


testing.run_module(__name__, __file__)

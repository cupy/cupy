import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestReshape(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 2, 6)).astype(numpy.float32)

    def check_forward(self, x_data):
        shape = self.gy.shape
        x = chainer.Variable(x_data)
        y = functions.reshape(x, shape)
        self.assertTrue((self.x.reshape(shape) == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = functions.reshape(x, self.gy.shape)
        y.grad = y_grad
        y.backward()

        shape = self.x.shepe
        self.assertTrue((self.gy.reshape(shape) == cuda.to_cpu(x.grad)).all())

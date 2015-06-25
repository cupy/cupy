from unittest import TestCase

import numpy
from chainer import Variable, cuda
from chainer.functions import reshape
from chainer.testing import attr

if cuda.available:
    cuda.init()


class TestReshape(TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 2, 6)).astype(numpy.float32)

    def check_forward(self, x_data):
        shape = self.gy.shape
        x = Variable(x_data)
        y = reshape(x, shape)
        self.assertTrue((self.x.reshape(shape) == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = reshape(x, self.gy.shape)
        y.grad = y_grad
        y.backward()

        shape = self.x.shepe
        self.assertTrue((self.gy.reshape(shape) == cuda.to_cpu(x.grad)).all())

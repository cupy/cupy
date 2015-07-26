import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


# TODO(Kenta OONO): This test fixture check types only. Add more detailed test.
class TestDropout(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_type_forward(self, x_data):
        x = chainer.Variable(x_data)
        functions.dropout(x)

    def test_type_forward_cpu(self):
        self.check_type_forward(self.x)

    @attr.gpu
    def test_type_forward_gpu(self):
        self.check_type_forward(cuda.to_gpu(self.x))

    def check_type_backward(self, x_data, gy_data):
        x = chainer.Variable(x_data)
        y = functions.dropout(x)
        y.grad = gy_data
        y.backward()

    def test_type_backward_cpu(self):
        self.check_type_backward(self.x, self.gy)

    @attr.gpu
    def test_type_backward_gpu(self):
        self.check_type_backward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

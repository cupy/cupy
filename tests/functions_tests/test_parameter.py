import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestParameter(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.gW = numpy.random.uniform(-1, 1,
                                       self.W.shape).astype(numpy.float32)
        self.func = functions.Parameter(self.W)

    def tearDown(self):
        del self.func

    def to_gpu(self):
        self.func.to_gpu()

    def check_forward(self):
        y = self.func()
        self.assertEqual(y.data.dtype, numpy.float32)
        self.assertTrue((self.W == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward()

    @attr.gpu
    def test_forward_gpu(self):
        self.to_gpu()
        self.check_forward()

    def check_backward(self, y_grad):
        self.func.gW.fill(0)
        y = self.func()
        y.grad = y_grad
        y.backward()
        self.assertTrue(
            (cuda.to_cpu(y_grad) == cuda.to_cpu(self.func.gW)).all())

    def test_backward_cpu(self):
        self.check_backward(self.gW)

    @attr.gpu
    def test_backward_gpu(self):
        self.to_gpu()
        self.check_backward(cuda.to_gpu(self.gW))

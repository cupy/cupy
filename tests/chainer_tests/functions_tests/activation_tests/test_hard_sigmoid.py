import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestHardSigmoid(unittest.TestCase):

    shape = (3, 4)

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.hard_sigmoid(x)
        expect = numpy.minimum(1.0, numpy.maximum(0.0, self.x * 0.2 + 0.5))
        gradient_check.assert_allclose(y.data, expect)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, grad):
        gradient_check.check_backward(
            functions.HardSigmoid(), x_data, grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

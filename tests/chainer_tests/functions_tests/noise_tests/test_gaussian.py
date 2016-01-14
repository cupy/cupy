import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.m = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.v = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_backward(self, m_data, v_data, y_grad):
        gradient_check.check_backward(
            functions.Gaussian(), (m_data, v_data), y_grad,
            atol=1e-4, rtol=1e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.m, self.v, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.m),
                            cuda.to_gpu(self.v),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        for i in range(self.x.size):
            if -0.01 < self.x.flat[i] < 0.01:
                self.x.flat[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.ReLU(use_cudnn), x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_cpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


class TestReLUZeroDim(TestReLU):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)


testing.run_module(__name__, __file__)

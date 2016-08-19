import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestZoneout(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        h_next = functions.zoneout(h, x, ratio=0)
        testing.assert_allclose(h_next.data, x.data)
        h_next = functions.zoneout(h, x, ratio=1.0)
        testing.assert_allclose(h_next.data, h.data)

    def check_backward(self, h_data, x_data, y_grad):
        gradient_check.check_backward(
            functions.Zoneout(0.0), (h_data, x_data), y_grad,
            atol=1e-4, rtol=1e-3)

        gradient_check.check_backward(
            functions.Zoneout(1.0), (h_data, x_data), y_grad,
            atol=1e-4, rtol=1e-3)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.h, self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.h),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)

import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import testing
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

    def check_forward(self, volatile):
        y = self.func(volatile)
        self.assertEqual(y.data.dtype, numpy.float32)
        self.assertTrue((self.W == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(False)

    def test_forward_cpu_volatile(self):
        self.check_forward(True)

    @attr.gpu
    def test_forward_gpu(self):
        self.to_gpu()
        self.check_forward(False)

    @attr.gpu
    def test_forward_gpu_volatile(self):
        self.to_gpu()
        self.check_forward(True)

    def check_backward(self, y_grad, volatile):
        self.func.gW.fill(0)
        y = self.func(volatile)
        y.grad = y_grad
        y.backward()
        if volatile:
            self.assertTrue(
                (cuda.to_cpu(self.func.gW) == numpy.zeros_like(y_grad)).all())
        else:
            self.assertTrue(
                (cuda.to_cpu(self.func.gW) == cuda.to_cpu(y_grad)).all())

    def test_backward_cpu(self):
        self.check_backward(self.gW, False)

    def test_backward_cpu_volatile(self):
        self.check_backward(self.gW, True)

    @attr.gpu
    def test_backward_gpu(self):
        self.to_gpu()
        self.check_backward(cuda.to_gpu(self.gW), False)

    @attr.gpu
    def test_backward_gpu_volatile(self):
        self.to_gpu()
        self.check_backward(cuda.to_gpu(self.gW), True)


class TestVolatile(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.func = functions.Parameter(self.W)

    def tearDown(self):
        del self.func

    def check_volatile(self, volatile):
        y = self.func(volatile=volatile)
        self.assertEqual(y.volatile, volatile)

    def test_volatile_cpu_volatile(self):
        self.check_volatile(True)

    def test_volatile_cpu_not_volatile(self):
        self.check_volatile(False)

    @attr.gpu
    def test_volatile_gpu_volatile(self):
        self.func.to_gpu()
        self.check_volatile(True)

    @attr.gpu
    def test_volatile_gpu_not_volatile(self):
        self.func.to_gpu()
        self.check_volatile(False)


testing.run_module(__name__, __file__)

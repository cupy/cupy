import unittest

import numpy

from chainer import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


class TestParameter(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.gW = numpy.random.uniform(-1, 1,
                                       self.W.shape).astype(numpy.float32)
        self.link = links.Parameter(self.W)
        self.link.zerograds()

    def tearDown(self):
        del self.link

    def to_gpu(self):
        self.link.to_gpu()

    def check_forward(self, volatile):
        y = self.link(volatile)
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
        y = self.link(volatile)
        y.grad = y_grad
        y.backward()
        if volatile:
            self.assertTrue(
                (cuda.to_cpu(self.link.W.grad) ==
                 numpy.zeros(y_grad.shape, dtype=y_grad.dtype)).all())
        else:
            self.assertTrue(
                (cuda.to_cpu(self.link.W.grad) == cuda.to_cpu(y_grad)).all())

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
        self.link = links.Parameter(self.W)

    def tearDown(self):
        del self.link

    def check_volatile(self, volatile):
        y = self.link(volatile=volatile)
        self.assertEqual(y.volatile, volatile)

    def test_volatile_cpu_volatile(self):
        self.check_volatile(True)

    def test_volatile_cpu_not_volatile(self):
        self.check_volatile(False)

    @attr.gpu
    def test_volatile_gpu_volatile(self):
        self.link.to_gpu()
        self.check_volatile(True)

    @attr.gpu
    def test_volatile_gpu_not_volatile(self):
        self.link.to_gpu()
        self.check_volatile(False)


class TestInit(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.p = links.Parameter(self.W)

    def check(self, p, xp):
        self.assertIsInstance(p.W.data, xp.ndarray)
        self.assertIsInstance(p.W.grad, xp.ndarray)

    def test_cpu(self):
        self.check(self.p, numpy)

    @attr.gpu
    def test_gpu_initialize_by_numpy_ndarray(self):
        self.p.to_gpu()
        self.check(self.p, cuda.cupy)

    @attr.gpu
    def test_gpu_initialize_by_cupy_ndarray(self):
        W = cuda.to_gpu(self.W)
        p = links.Parameter(W)
        self.check(p, cuda.cupy)


testing.run_module(__name__, __file__)

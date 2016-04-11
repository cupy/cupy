import unittest

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
import numpy


class TestPrintHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.PrintHook()
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_name(self):
        self.assertEqual(self.h.name, 'PrintHook')

    def test_forward_cpu(self):
        with self.h:
            self.l(chainer.Variable(self.x))

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        with self.h:
            self.l(chainer.Variable(cuda.to_gpu(self.x)))

    def test_backward_cpu(self):
        with self.h:
            gradient_check.check_backward(self.l, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        with self.h:
            gradient_check.check_backward(
                self.l, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestPrintHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.PrintHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_forward_cpu(self):
        self.f(chainer.Variable(self.x))

    @attr.gpu
    def test_fowward_gpu(self):
        self.f(chainer.Variable(cuda.to_gpu(self.x)))

    def test_backward_cpu(self):
        gradient_check.check_backward(self.f, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        gradient_check.check_backward(
            self.f, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

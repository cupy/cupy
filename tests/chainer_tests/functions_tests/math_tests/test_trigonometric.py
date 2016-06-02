import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class UnaryFunctionsTestBase(object):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.x, self.gy = self.make_data()

    def check_forward(self, op, op_np, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        testing.assert_allclose(
            op_np(self.x), y.data, atol=1e-7, rtol=1e-7)

    def check_forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    def check_forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_cos_forward_cpu(self):
        self.check_forward_cpu(F.cos, numpy.cos)

    @condition.retry(3)
    def test_sin_forward_cpu(self):
        self.check_forward_cpu(F.sin, numpy.sin)

    @attr.gpu
    @condition.retry(3)
    def test_cos_forward_gpu(self):
        self.check_forward_gpu(F.cos, numpy.cos)

    @attr.gpu
    @condition.retry(3)
    def test_sin_forward_gpu(self):
        self.check_forward_gpu(F.sin, numpy.sin)

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(op, x_data, y_grad)

    def check_backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def check_backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @condition.retry(3)
    def test_cos_backward_cpu(self):
        self.check_backward_cpu(F.cos)

    @condition.retry(3)
    def test_sin_backward_cpu(self):
        self.check_backward_cpu(F.sin)

    @attr.gpu
    @condition.retry(3)
    def test_cos_backward_gpu(self):
        self.check_backward_gpu(F.cos)

    @attr.gpu
    @condition.retry(3)
    def test_sin_backward_gpu(self):
        self.check_backward_gpu(F.sin)

    def test_sin(self):
        self.assertEqual(F.Sin().label, 'sin')

    def test_cos(self):
        self.assertEqual(F.Cos().label, 'cos')


class TestUnaryFunctionsSimple(UnaryFunctionsTestBase, unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        return x, gy


class TestUnaryFunctionsZeroDimension(UnaryFunctionsTestBase,
                                      unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        return x, gy


testing.run_module(__name__, __file__)

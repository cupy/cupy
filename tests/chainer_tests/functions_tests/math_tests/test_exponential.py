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
    def test_exp_forward_cpu(self):
        self.check_forward_cpu(F.exp, numpy.exp)

    @condition.retry(3)
    def test_log_forward_cpu(self):
        self.check_forward_cpu(F.log, numpy.log)

    @attr.gpu
    @condition.retry(3)
    def test_exp_forward_gpu(self):
        self.check_forward_gpu(F.exp, numpy.exp)

    @attr.gpu
    @condition.retry(3)
    def test_log_forward_gpu(self):
        self.check_forward_gpu(F.log, numpy.log)

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(op, x_data, y_grad)

    def check_backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def check_backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @condition.retry(3)
    def test_exp_backward_cpu(self):
        self.check_backward_cpu(F.exp)

    @condition.retry(3)
    def test_log_backward_cpu(self):
        self.check_backward_cpu(F.log)

    @attr.gpu
    @condition.retry(3)
    def test_exp_backward_gpu(self):
        self.check_backward_gpu(F.exp)

    @attr.gpu
    @condition.retry(3)
    def test_log_backward_gpu(self):
        self.check_backward_gpu(F.log)

    def test_exp(self):
        self.assertEqual(F.Exp().label, 'exp')

    def test_log(self):
        self.assertEqual(F.Log().label, 'log')


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

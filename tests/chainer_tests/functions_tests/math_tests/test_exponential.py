import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class UnaryFunctionsTestBase(unittest.TestCase):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.x, self.gy = self.make_data()

    def check_forward(self, op, op_np, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        self.assertEqual(x.data.dtype, y.data.dtype)
        testing.assert_allclose(
            op_np(self.x), y.data, atol=1e-7, rtol=1e-7)

    def check_forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    def check_forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, cuda.to_gpu(self.x))

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(
            op, x_data, y_grad, atol=1e-4, rtol=1e-3, dtype=numpy.float64)

    def check_backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def check_backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_label(self, op, expected):
        self.assertEqual(op().label, expected)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestExp(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.exp, numpy.exp)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.exp, numpy.exp)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.exp)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.exp)

    def test_label(self):
        self.check_label(F.Exp, 'exp')


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLog(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.log, numpy.log)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.log, numpy.log)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.log)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.log)

    def test_label(self):
        self.check_label(F.Log, 'log')


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLog2(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.log2, numpy.log2)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.log2, numpy.log2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.log2)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.log2)

    def test_label(self):
        self.check_label(F.Log2, 'log2')


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLog10(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.log10, numpy.log10)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.log10, numpy.log10)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.log10)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.log10)

    def test_label(self):
        self.check_label(F.Log10, 'log10')


testing.run_module(__name__, __file__)

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

    def check_forward(self, op, op_xp, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        self.assertEqual(x.data.dtype, y.data.dtype)
        testing.assert_allclose(op_xp(x_data), y.data, atol=1e-7, rtol=1e-7)

    def check_forward_cpu(self, op, op_xp):
        self.check_forward(op, op_xp, self.x)

    def check_forward_gpu(self, op, op_xp):
        self.check_forward(op, op_xp, cuda.to_gpu(self.x))

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
class TestCosh(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.cosh, numpy.cosh)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.cosh, cuda.cupy.cosh)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.cosh)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.cosh)

    def test_label(self):
        self.check_label(F.Cosh, 'cosh')


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSinh(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.sinh, numpy.sinh)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.sinh, cuda.cupy.sinh)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.sinh)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.sinh)

    def test_label(self):
        self.check_label(F.Sinh, 'sinh')


testing.run_module(__name__, __file__)

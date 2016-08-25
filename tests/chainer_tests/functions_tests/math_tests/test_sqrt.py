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
        testing.assert_allclose(
            op_xp(x_data, dtype=x_data.dtype), y.data, atol=1e-7, rtol=1e-7)

    def check_forward_cpu(self, op, op_xp):
        self.check_forward(op, op_xp, self.x)

    def check_forward_gpu(self, op, op_xp):
        self.check_forward(op, op_xp, cuda.to_gpu(self.x))

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(
            op, x_data, y_grad, atol=5e-4, rtol=5e-3, dtype=numpy.float64)

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
class TestSqrt(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.sqrt, numpy.sqrt)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.sqrt, cuda.cupy.sqrt)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.sqrt)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.sqrt)

    def test_label(self):
        self.check_label(F.Sqrt, 'sqrt')


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRsqrt(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        def np_rsqrt(x, dtype=numpy.float32):
            return numpy.reciprocal(numpy.sqrt(x, dtype=dtype))
        self.check_forward_cpu(F.rsqrt, np_rsqrt)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        def cp_rsqrt(x, dtype=numpy.float32):
            return cuda.cupy.reciprocal(cuda.cupy.sqrt(x, dtype=dtype))
        self.check_forward_gpu(F.rsqrt, cp_rsqrt)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.rsqrt)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.rsqrt)


testing.run_module(__name__, __file__)

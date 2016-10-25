import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'in_shape': [(5, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSquaredDifference(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)

    def check_forward(self, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.squared_difference(x1, x2)
        y_expect = (self.x1 - self.x2) ** 2
        testing.assert_allclose(y.data, y_expect, atol=1e-3)
        self.assertEqual(y.data.shape, self.in_shape)
        self.assertEqual(y.data.dtype, self.dtype)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1, x2, g_data):
        x_data = (x1, x2)
        gradient_check.check_backward(
            functions.SquaredDifference(),
            x_data, g_data, dtype=numpy.float64, atol=1e-2, rtol=2e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.g)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        g = cuda.to_gpu(self.g)
        self.check_backward(x1, x2, g)


testing.run_module(__name__, __file__)

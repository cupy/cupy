import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(), (3, 2)],
}))
class Log1pFunctionTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = F.log1p(x)
        testing.assert_allclose(
            numpy.log1p(self.x), y.data, atol=1e-7, rtol=1e-7)

    @condition.retry(3)
    def test_log1p_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_log1p_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(F.log1p, x_data, y_grad)

    @condition.retry(3)
    def test_log1p_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_log1p_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_log1p(self):
        self.assertEqual(F.Log1p().label, 'log1p')


testing.run_module(__name__, __file__)

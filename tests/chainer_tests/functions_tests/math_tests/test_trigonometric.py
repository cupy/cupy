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
    'func_name': ['cos', 'sin', 'tan'],
    'shape': [(3, 2), ()],
}))
class UnaryFunctionsTest(unittest.TestCase):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.func = getattr(F, self.func_name)
        camel_name = self.func_name[0].upper() + self.func_name[1:]
        self.func_class = getattr(F, camel_name)
        self.np_func = getattr(numpy, self.func_name)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        testing.assert_allclose(
            self.np_func(self.x), y.data, atol=1e-7, rtol=1e-7)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self.func, x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_label(self):
        self.assertEqual(self.func_class().label, self.func_name)


testing.run_module(__name__, __file__)

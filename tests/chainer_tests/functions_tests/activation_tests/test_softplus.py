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
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSoftplus(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.beta = numpy.random.uniform(1, 2, ())
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.softplus(x, beta=self.beta)
        x_value = cuda.to_cpu(x_data)
        y_exp = numpy.log(1 + numpy.exp(self.beta * x_value)) / self.beta
        self.assertEqual(y.data.dtype, self.dtype)
        testing.assert_allclose(
            y_exp, y.data, **self.check_forward_options)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Softplus(beta=self.beta), x_data, y_grad,
            **self.check_backward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

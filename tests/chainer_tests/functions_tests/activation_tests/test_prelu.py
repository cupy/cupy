import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.activation import prelu
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (1,)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestPReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if -0.05 < self.x[i] < 0.05:
                self.x[i] = 0.5
        self.W = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        y = functions.prelu(x, W)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                y_expect[i] *= self.W

        testing.assert_allclose(
            y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.W)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W))

    def check_backward(self, x_data, W_data, y_grad):
        gradient_check.check_backward(
            prelu.PReLUFunction(), (x_data, W_data), y_grad,
            **self.check_backward_options)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

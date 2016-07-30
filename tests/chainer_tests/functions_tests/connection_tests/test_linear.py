import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import linear
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestNonparameterizedLinear(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, (2, 3)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, 2).astype(self.x_dtype)

        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(-1, 1, (4, 2)).astype(self.x_dtype)
        self.y = self.x.dot(self.W.T) + self.b
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data, b_data, y_expect):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        if b_data is None:
            y = functions.linear(x, W)
        else:
            b = chainer.Variable(b_data)
            y = functions.linear(x, W, b)
        self.assertEqual(y.data.dtype, self.x_dtype)
        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b,
                           self.x.dot(self.W.T) + self.b)

    @condition.retry(3)
    def test_forward_cpu_nobias(self):
        self.check_forward(self.x, self.W, None, self.x.dot(self.W.T))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b),
            cuda.to_gpu(self.x.dot(self.W.T) + self.b))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_nobias(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), None,
            cuda.to_gpu(self.x.dot(self.W.T)))

    def check_backward(self, x_data, W_data, b_data, y_grad):
        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            linear.LinearFunction(), args, y_grad,
            eps=1e-2, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

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
    'shape': [(3, 4), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLinearInterpolate(unittest.TestCase):

    def setUp(self):
        self.p = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.y = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, p_data, x_data, y_data):
        p = chainer.Variable(p_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        z = functions.linear_interpolate(p, x, y)
        self.assertEqual(z.data.dtype.type, self.dtype)
        expect = self.p * self.x + (1 - self.p) * self.y
        testing.assert_allclose(
            z.data, expect, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.p, self.x, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.p),
                           cuda.to_gpu(self.x),
                           cuda.to_gpu(self.y))

    def check_backward(self, p_data, x_data, y_data, grad):
        gradient_check.check_backward(
            functions.LinearInterpolate(), (p_data, x_data, y_data), grad,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.p, self.x, self.y, self.g)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.p),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.y),
                            cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

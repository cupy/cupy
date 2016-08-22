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
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestHardSigmoid(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Avoid unstability of numerical grad
        for i in numpy.ndindex(self.shape):
            if -0.35 < self.x[i] < 0.15 or 0.15 < self.x[i] < 0.35:
                self.x[i] = 0.0

        self.check_forward_option = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype is numpy.float16:
            self.check_forward_option = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.hard_sigmoid(x)
        self.assertIs(y.data.dtype, x_data.dtype)
        expect = numpy.minimum(1.0, numpy.maximum(0.0, self.x * 0.2 + 0.5))
        testing.assert_allclose(
            y.data, expect, **self.check_forward_option)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, grad):
        gradient_check.check_backward(
            functions.HardSigmoid(), x_data, grad,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'in_shape': [(5, 2)],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
class TestSquaredDifference(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.squared_difference(x1, x2)
        self.assertEqual(y.data.dtype, self.dtype)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.squared_difference(x1, x2)
        y.grad = y.data
        y.backward()
        testing.assert_allclose(x1.data, x1.grad, atol=0, rtol=0)
        testing.assert_allclose(x2.data, x2.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.x)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)

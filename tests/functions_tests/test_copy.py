import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check
from chainer import testing


class Copy(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(
            -1, 1, (10, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)

    def test_check_forward_cpu(self):
        x = chainer.Variable(self.x_data)
        y = functions.copy(x, -1)
        gradient_check.assert_allclose(self.x_data, y.data, atol=0, rtol=0)

    def test_check_backward_cpu(self):
        x = chainer.Variable(self.x_data)
        y = functions.copy(x, -1)
        y.grad = self.gy
        y.backward()
        gradient_check.assert_allclose(x.grad, self.gy, atol=0, rtol=0)


testing.run_module(__name__, __file__)

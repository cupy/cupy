import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.mean_squared_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.
        for i in numpy.ndindex(self.x0.shape):
            loss_expect += (self.x0[i] - self.x1[i]) ** 2
        loss_expect /= self.x0.size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.success_at_least(3, 1)
    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    @condition.success_at_least(3, 1)
    def test_forwrad_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.mean_squared_error(x0, x1)
        loss.backward()

        func = loss.creator
        f = lambda: func.forward((x0.data, x1.data))
        gx0, gx1 = gradient_check.numerical_grad(
            f, (x0.data, x1.data), (1,), eps=1e-2)

        gradient_check.assert_allclose(gx0, x0.grad)
        gradient_check.assert_allclose(gx1, x1.grad)
        self.assertEqual(x0.grad.dtype, numpy.float32)
        self.assertEqual(x1.grad.dtype, numpy.float32)

    @condition.success_at_least(3, 1)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1)

    @attr.gpu
    @condition.success_at_least(3, 1)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

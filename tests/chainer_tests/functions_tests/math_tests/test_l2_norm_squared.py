import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.math.l2_norm_squared import _as_two_dim
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestL2NormSquared(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, (4, 3, 5)).astype(numpy.float32)
        self.x0 = self.x1.reshape((len(self.x1), -1))
        self.gy = numpy.random.uniform(-1, 1, (4,)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.l2_norm_squared(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.empty(len(self.x0))
        for n in six.moves.range(len(self.x0)):
            y_expect[n] = sum(map(lambda x: x * x, self.x0[n]))

        gradient_check.assert_allclose(y_expect, y_data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x0)

    @condition.retry(3)
    def test_forward_cpu_three_dim(self):
        self.check_forward(self.x1)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_three_dim(self):
        self.check_forward(cuda.to_gpu(self.x1))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(_as_two_dim(x_data))
        y = functions.l2_norm_squared(x)

        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,), eps=1)

        gradient_check.assert_allclose(gx, x.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.gy))

    @condition.retry(3)
    def test_backward_cpu_three_dim(self):
        self.check_backward(self.x1, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_three_dim(self):
        self.check_backward(cuda.to_gpu(self.x1), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

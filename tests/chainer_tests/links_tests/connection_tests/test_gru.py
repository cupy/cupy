import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def _sigmoid(x):
    xp = cuda.get_array_module(x)
    return 1 / (1 + xp.exp(-x))


def _gru(func, h, x):
    xp = cuda.get_array_module(h, x)

    r = _sigmoid(x.dot(func.W_r.W.data.T) + h.dot(func.U_r.W.data.T))
    z = _sigmoid(x.dot(func.W_z.W.data.T) + h.dot(func.U_z.W.data.T))
    h_bar = xp.tanh(x.dot(func.W.W.data.T) + (r * h).dot(func.U.W.data.T))
    y = (1 - z) * h + z * h_bar
    return y


class TestGRU(unittest.TestCase):

    def setUp(self):
        self.link = links.GRU(8)
        self.x = numpy.random.uniform(-1, 1, (3, 8)).astype(numpy.float32)
        self.h = numpy.random.uniform(-1, 1, (3, 8)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 8)).astype(numpy.float32)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        y = self.link(h, x)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = _gru(self.link, h_data, x_data)
        gradient_check.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.h),
                           cuda.to_gpu(self.x))

    def check_backward(self, h_data, x_data, y_grad):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        y = self.link(h, x)

        y.grad = y_grad
        y.backward()

        f = lambda: (_gru(self.link, h.data, x.data),)
        gh, gx = gradient_check.numerical_grad(f, (h.data, x.data), (y.grad,))

        gradient_check.assert_allclose(gh, h.grad, atol=1e-3)
        gradient_check.assert_allclose(gx, x.grad, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.h, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.h),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

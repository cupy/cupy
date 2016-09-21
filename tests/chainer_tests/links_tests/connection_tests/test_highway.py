import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestHighway(unittest.TestCase):

    in_out_size = 3

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (5, self.in_out_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (5, self.in_out_size)).astype(numpy.float32)
        self.link = links.Highway(
            self.in_out_size)

        Wh = self.link.plain.W.data
        Wh[...] = numpy.random.uniform(-1, 1, Wh.shape)
        bh = self.link.plain.b.data
        bh[...] = numpy.random.uniform(-1, 1, bh.shape)

        Wt = self.link.transform.W.data
        Wt[...] = numpy.random.uniform(-1, 1, Wt.shape)
        bt = self.link.transform.b.data
        bt[...] = numpy.random.uniform(-1, 1, bt.shape)
        self.link.cleargrads()

        self.Wh = Wh.copy()  # fixed on CPU
        self.bh = bh.copy()  # fixed on CPU
        self.Wt = Wt.copy()  # fixed on CPU
        self.bt = bt.copy()  # fixed on CPU

        a = self.relu(self.x.dot(Wh.T) + bh)
        b = self.sigmoid(self.x.dot(Wt.T) + bt)
        self.y = (a * b +
                  self.x * (numpy.ones_like(self.x) - b))

    def relu(self, x):
        return numpy.maximum(x, 0, dtype=x.dtype)

    def sigmoid(self, x):
        half = x.dtype.type(0.5)
        return numpy.tanh(x * half) * half + half

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        testing.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad,
            (self.link.plain.W, self.link.plain.b,
             self.link.transform.W, self.link.transform.b),
            eps=1e-2, atol=3.2e-3, rtol=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr



class TestHighway(unittest.TestCase):

    in_out_size = 3

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_out_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (10, self.in_out_size)).astype(numpy.float32)
        self.link = links.Highway(
            self.in_out_size)

        Wh = self.link.plain.W.data
        bh = self.link.plain.b.data

        Wt = self.link.transform.W.data
        bt = self.link.transform.b.data

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
        gradient_check.assert_allclose(self.y, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.link(x)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

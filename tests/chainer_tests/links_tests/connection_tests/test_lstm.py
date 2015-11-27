import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'in_size': 10, 'out_size': 10},
    {'in_size': 10, 'out_size': 40},
)
class TestLSTM(unittest.TestCase):

    def setUp(self):
        self.link = links.LSTM(self.in_size, self.out_size)
        upward = self.link.upward.W.data
        upward[...] = numpy.random.uniform(-1, 1, upward.shape)
        lateral = self.link.lateral.W.data
        lateral[...] = numpy.random.uniform(-1, 1, lateral.shape)
        self.link.zerograds()

        self.upward = upward.copy()  # fixed on CPU
        self.lateral = lateral.copy()  # fixed on CPU

        x_shape = (4, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        xp = self.link.xp
        x = chainer.Variable(x_data)
        h1 = self.link(x)
        c0 = chainer.Variable(xp.zeros((len(self.x), self.out_size),
                                       dtype=self.x.dtype))
        c1_expect, h1_expect = functions.lstm(c0, self.link.upward(x))
        gradient_check.assert_allclose(h1.data, h1_expect.data)
        gradient_check.assert_allclose(self.link.h.data, h1_expect.data)
        gradient_check.assert_allclose(self.link.c.data, c1_expect.data)

        h2 = self.link(x)
        c2_expect, h2_expect = \
            functions.lstm(c1_expect,
                           self.link.upward(x) + self.link.lateral(h1))
        gradient_check.assert_allclose(h2.data, h2_expect.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

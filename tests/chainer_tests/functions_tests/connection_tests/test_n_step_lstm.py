import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions


class TestNStepLSTM(unittest.TestCase):

    length = 5
    n_batch = 4
    in_size = 3
    out_size = 3

    def setUp(self):
        self.rnn = functions.NStepLSTM()
        x_shape = (self.length, 1, self.n_batch, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        h_shape = (self.length, self.n_batch, self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(self, h_data, c_data, x_data):
        h = chainer.Variable(h_data)
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        self.rnn(h, c, x)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           cuda.to_gpu(self.x))

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


class TestNStepLSTM(unittest.TestCase):

    length = 3
    n_batch = 2
    in_size = 4
    out_size = 4
    n_layers = 2
    dropout = 0.0
    seed = 1337

    def setUp(self):
        handle = cuda.cupy.cudnn.get_handle()
        states = functions.n_step_lstm.DropoutStates.create(handle, self.dropout, self.seed)
        self.rnn = functions.NStepLSTM(self.n_layers, states)

        x_shape = (self.length, self.n_batch, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        h_shape = (self.n_layers, self.n_batch, self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

        w_shape = (self.n_layers, 8, self.out_size, self.out_size)
        self.w = numpy.random.uniform(-1, 1, w_shape).astype(numpy.float32)
        b_shape = (self.n_layers, 8, self.out_size)
        self.b = numpy.random.uniform(-1, 1, b_shape).astype(numpy.float32)

        self.dy = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(self, h_data, c_data, x_data, w_data, b_data):
        h = chainer.Variable(h_data)
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        w = chainer.Variable(w_data)
        b = chainer.Variable(b_data)
        hy, cy, y = self.rnn(h, c, x, w, b)

        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for xi in self.x:
            x = xi
            for layer in range(self.n_layers):
                w = self.w[layer]
                b = self.b[layer]
                h_prev = e_hy[layer]
                c_prev = e_cy[layer]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer] = e_h
                e_cy[layer] = e_c

                x = e_h

        gradient_check.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        gradient_check.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           cuda.to_gpu(self.x),
                           cuda.to_gpu(self.w),
                           cuda.to_gpu(self.b))

    def check_backward(self, h_data, c_data, x_data, w_data, b_data,
                       dhy_data, dcy_data, dy_data):
        gradient_check.check_backward(
            self.rnn,
            (h_data, c_data, x_data, w_data, b_data),
            (dhy_data, dcy_data, dy_data), eps=1e-2, rtol=1e-4, atol=1e-4)

    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.hx),
                            cuda.to_gpu(self.cx),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.w),
                            cuda.to_gpu(self.b),
                            cuda.to_gpu(self.dhy),
                            cuda.to_gpu(self.dcy),
                            cuda.to_gpu(self.dy))

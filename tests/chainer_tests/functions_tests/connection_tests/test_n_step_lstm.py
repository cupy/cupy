import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


class TestNStepLSTM(unittest.TestCase):

    batches = [4, 3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 4
    n_layers = 2
    dropout = 0.0
    seed = 1337

    def setUp(self):
        handle = cuda.cupy.cudnn.get_handle()
        states = functions.n_step_lstm.DropoutStates.create(handle, self.dropout, self.seed)
        self.rnn = functions.NStepLSTM(self.n_layers, states)

        self.xs = [numpy.random.uniform(-1, 1, (b, self.in_size)).astype('f')
                   for b in self.batches]
        h_shape = (self.n_layers, self.batches[0], self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                self.ws.append(numpy.random.uniform(
                    -1, 1, (self.out_size, w_in)).astype('f'))
                self.bs.append(numpy.random.uniform(
                    -1, 1, (self.out_size,)).astype('f'))

        self.dys = [numpy.random.uniform(-1, 1, (b, self.out_size)).astype('f')
                    for b in self.batches]
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(self, h_data, c_data, xs_data, ws_data, bs_data):
        h = chainer.Variable(h_data)
        c = chainer.Variable(c_data)
        xs = [chainer.Variable(x_data) for x_data in xs_data]
        ws = [chainer.Variable(w_data) for w_data in ws_data]
        bs = [chainer.Variable(b_data) for b_data in bs_data]
        ret = self.rnn(*([h, c] + ws + bs + xs))
        hy, cy = ret[:2]
        ys = ret[2:]

        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for ind in range(self.length):
            x = self.xs[ind]
            ey = None
            for layer in range(self.n_layers):
                batch = x.shape[0]
                w = self.ws[layer * 8: layer * 8 + 8]
                b = self.bs[layer * 8: layer * 8 + 8]
                h_prev = e_hy[layer, :batch]
                c_prev = e_cy[layer, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                if ey is None:
                    ey = e_h
                else:
                    ey[:batch] = e_h
                e_hy[layer, :batch] = e_h
                e_cy[layer, :batch] = e_c

                x = e_h

            gradient_check.assert_allclose(ys[ind].data, ey)

        gradient_check.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        gradient_check.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [cuda.to_gpu(w) for w in self.ws],
                           [cuda.to_gpu(b) for b in self.bs])

    def check_backward(self, h_data, c_data, xs_data, ws_data, bs_data,
                       dhy_data, dcy_data, dys_data):
        gradient_check.check_backward(
            self.rnn,
            tuple([h_data, c_data] + ws_data + bs_data + xs_data),
            tuple([dhy_data, dcy_data] + dys_data),
            eps=1e-2, rtol=1e-4, atol=1e-4)

    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.hx),
                            cuda.to_gpu(self.cx),
                            [cuda.to_gpu(x) for x in self.xs],
                            [cuda.to_gpu(w) for w in self.ws],
                            [cuda.to_gpu(b) for b in self.bs],
                            cuda.to_gpu(self.dhy),
                            cuda.to_gpu(self.dcy),
                            [cuda.to_gpu(dy) for dy in self.dys])

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
class TestNStepLSTM(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0
    seed = 1337

    def setUp(self):
        shape = (self.n_layer, len(self.lengths), self.out_size)
        self.h = numpy.random.uniform(-1, 1, shape).astype('f')
        self.c = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gc = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepLSTM(
            self.n_layer, self.in_size, self.out_size, self.dropout, self.seed,
            use_cudnn=self.use_cudnn)

        for layer in self.rnn:
            for p in layer.params():
                p.data[...] = numpy.random.uniform(-1, 1, p.data.shape)
        self.rnn.zerograds()

    def check_forward(self, h_data, c_data, xs_data):
        h = chainer.Variable(h_data)
        c = chainer.Variable(c_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, cy, ys = self.rnn(h, c, xs)

        self.assertEqual(hy.data.shape, h.data.shape)
        self.assertEqual(cy.data.shape, c.data.shape)
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            self.assertEqual(len(x.data), len(y.data))
            self.assertEqual(y.data.shape[1], self.out_size)

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                p = self.rnn[layer]
                h_prev = self.h[layer, batch]
                c_prev = self.c[layer, batch]
                hs = []
                for x in seq:
                    i = sigmoid(x.dot(p.w0.data.T) + h_prev.dot(p.w4.data.T) +
                                p.b0.data + p.b4.data)
                    f = sigmoid(x.dot(p.w1.data.T) + h_prev.dot(p.w5.data.T) +
                                p.b1.data + p.b5.data)
                    c_bar = numpy.tanh(
                        x.dot(p.w2.data.T) + h_prev.dot(p.w6.data.T) +
                        p.b2.data + p.b6.data)
                    o = sigmoid(x.dot(p.w3.data.T) + h_prev.dot(p.w7.data.T) +
                                p.b3.data + p.b7.data)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs.append(e_h)

                seq = hs
                gradient_check.assert_allclose(hy.data[layer, batch], h_prev)
                gradient_check.assert_allclose(cy.data[layer, batch], c_prev)

            for y, ey in zip(ys[batch].data, seq):
                gradient_check.assert_allclose(y, ey)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.rnn.to_gpu()
        self.check_forward(
            cuda.to_gpu(self.h),
            cuda.to_gpu(self.c),
            [cuda.to_gpu(x) for x in self.xs])

    def check_backward(
            self, h_data, c_data, xs_data, gh_data, gc_data, gys_data):

        def fun(*args):
            h, c = args[:2]
            xs = args[2:]
            hy, cy, ys = self.rnn(h, c, xs)
            return tuple([hy, cy] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        gradient_check.check_backward(
            fun, tuple([h_data, c_data] + xs_data),
            tuple([gh_data, gc_data] + gys_data),
            tuple(params), eps=1e-2, rtol=1e-2)

    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.c, self.xs, self.gh, self.gc, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        self.check_backward(
            cuda.to_gpu(self.h),
            cuda.to_gpu(self.c),
            [cuda.to_gpu(x) for x in self.xs],
            cuda.to_gpu(self.gh),
            cuda.to_gpu(self.gc),
            [cuda.to_gpu(gy) for gy in self.gys])


testing.run_module(__name__, __file__)

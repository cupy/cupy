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
    'hidden_none': [True, False],
}))
class TestNStepGRU(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layer, len(self.lengths), self.out_size)
        if self.hidden_none:
            self.h = numpy.zeros(shape, 'f')
        else:
            self.h = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepGRU(
            self.n_layer, self.in_size, self.out_size, self.dropout,
            use_cudnn=self.use_cudnn)

        for layer in self.rnn:
            for p in layer.params():
                p.data[...] = numpy.random.uniform(-1, 1, p.data.shape)
        self.rnn.zerograds()

    def check_forward(self, h_data, xs_data):
        if self.hidden_none:
            h = None
        else:
            h = chainer.Variable(h_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, ys = self.rnn(h, xs)

        self.assertEqual(hy.data.shape, h_data.shape)
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            self.assertEqual(len(x.data), len(y.data))
            self.assertEqual(y.data.shape[1], self.out_size)

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                p = self.rnn[layer]
                h_prev = self.h[layer, batch]
                hs = []
                for x in seq:
                    # GRU
                    z = sigmoid(x.dot(p.w1.data.T) + h_prev.dot(p.w4.data.T) +
                                p.b1.data + p.b4.data)
                    r = sigmoid(x.dot(p.w0.data.T) + h_prev.dot(p.w3.data.T) +
                                p.b0.data + p.b3.data)
                    h_bar = numpy.tanh(x.dot(p.w2.data.T) +
                                       r * ((h_prev).dot(p.w5.data.T) +
                                            p.b5.data) + p.b2.data)
                    e_h = (1 - z) * h_bar + z * h_prev

                    h_prev = e_h
                    hs.append(e_h)

                seq = hs
                testing.assert_allclose(hy.data[layer, batch], h_prev)

            for y, ey in zip(ys[batch].data, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.rnn.to_gpu()
        self.check_forward(
            cuda.to_gpu(self.h),
            [cuda.to_gpu(x) for x in self.xs])

    def check_backward(
            self, h_data, xs_data, gh_data, gys_data):

        def fun(*args):
            if self.hidden_none:
                h = None
                xs = args
            else:
                h, = args[:1]
                xs = args[1:]
            hy, ys = self.rnn(h, xs)
            return tuple([hy, ] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        if self.hidden_none:
            in_data = xs_data
        else:
            in_data = [h_data, ] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data, ] + gys_data),
            tuple(params), eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.xs, self.gh, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        self.check_backward(
            cuda.to_gpu(self.h),
            [cuda.to_gpu(x) for x in self.xs],
            cuda.to_gpu(self.gh),
            [cuda.to_gpu(gy) for gy in self.gys])


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'hidden_none': [True, False],
}))
class TestNStepBiGRU(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layer * 2, len(self.lengths), self.out_size)
        if self.hidden_none:
            self.h = numpy.zeros(shape, 'f')
        else:
            self.h = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size*2)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepBiGRU(
            self.n_layer, self.in_size, self.out_size, self.dropout,
            use_cudnn=self.use_cudnn)

        for layer in self.rnn:
            for p in layer.params():
                p.data[...] = numpy.random.uniform(-1, 1, p.data.shape)
        self.rnn.zerograds()

    def check_forward(self, h_data, xs_data):
        if self.hidden_none:
            h = None
        else:
            h = chainer.Variable(h_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, ys = self.rnn(h, xs)

        self.assertEqual(hy.data.shape, h_data.shape)
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            self.assertEqual(len(x.data), len(y.data))
            self.assertEqual(y.data.shape[1], self.out_size * 2)

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                # forward
                di = 0
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                hs_f = []
                for x in seq:
                    # GRU
                    z = sigmoid(x.dot(p.w1.data.T) + h_prev.dot(p.w4.data.T) +
                                p.b1.data + p.b4.data)
                    r = sigmoid(x.dot(p.w0.data.T) + h_prev.dot(p.w3.data.T) +
                                p.b0.data + p.b3.data)
                    h_bar = numpy.tanh(x.dot(p.w2.data.T) +
                                       r * ((h_prev).dot(p.w5.data.T) +
                                            p.b5.data) + p.b2.data)
                    e_h = (1 - z) * h_bar + z * h_prev

                    h_prev = e_h
                    hs_f.append(e_h)

                testing.assert_allclose(hy.data[layer_idx, batch], h_prev)

                # backward
                di = 1
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                hs_b = []
                for x in reversed(seq):
                    # GRU
                    z = sigmoid(x.dot(p.w1.data.T) + h_prev.dot(p.w4.data.T) +
                                p.b1.data + p.b4.data)
                    r = sigmoid(x.dot(p.w0.data.T) + h_prev.dot(p.w3.data.T) +
                                p.b0.data + p.b3.data)
                    h_bar = numpy.tanh(x.dot(p.w2.data.T) +
                                       r * ((h_prev).dot(p.w5.data.T) +
                                            p.b5.data) + p.b2.data)
                    e_h = (1 - z) * h_bar + z * h_prev
                    h_prev = e_h
                    hs_b.append(e_h)
                testing.assert_allclose(hy.data[layer_idx, batch], h_prev)

                hs_b.reverse()
                seq = [numpy.concatenate([hfi, hbi], axis=0) for (hfi, hbi)
                       in zip(hs_f, hs_b)]

            for y, ey in zip(ys[batch].data, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.rnn.to_gpu()
        self.check_forward(
            cuda.to_gpu(self.h),
            [cuda.to_gpu(x) for x in self.xs])

    def check_backward(
            self, h_data, xs_data, gh_data, gys_data):

        def fun(*args):
            if self.hidden_none:
                h = None
                xs = args
            else:
                h, = args[:1]
                xs = args[1:]
            hy, ys = self.rnn(h, xs)
            return tuple([hy, ] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        if self.hidden_none:
            in_data = xs_data
        else:
            in_data = [h_data, ] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data, ] + gys_data),
            tuple(params), eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.xs, self.gh, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        self.check_backward(
            cuda.to_gpu(self.h),
            [cuda.to_gpu(x) for x in self.xs],
            cuda.to_gpu(self.gh),
            [cuda.to_gpu(gy) for gy in self.gys])

testing.run_module(__name__, __file__)

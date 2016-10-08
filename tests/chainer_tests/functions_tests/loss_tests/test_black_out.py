import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestBlackOut(unittest.TestCase):

    batch_size = 5
    in_size = 4
    n_vocab = 3
    n_samples = 2

    def setUp(self):
        x_shape = (self.batch_size, self.in_size)
        self.x = numpy.random.uniform(
            -1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(
            self.n_vocab, size=self.batch_size).astype(numpy.int32)
        w_shape = (self.n_vocab, self.in_size)
        self.W = numpy.random.uniform(
            -1, 1, w_shape).astype(numpy.float32)
        self.samples = numpy.random.randint(
            self.n_vocab, size=self.batch_size * self.n_samples) \
            .astype(numpy.int32).reshape((self.batch_size, self.n_samples))

    def check_forward(self, x_data, t_data, w_data, samples_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        w = chainer.Variable(w_data)
        samples = chainer.Variable(samples_data)

        y = functions.black_out(x, t, w, samples)

        expect_y = numpy.empty((self.batch_size), dtype=numpy.float32)
        for b in range(self.batch_size):
            z = 0
            for i in range(self.n_samples):
                w = self.samples[b, i]
                z += numpy.exp(self.W[w].dot(self.x[b]))
            y0 = self.W[self.t[b]].dot(self.x[b])
            z += numpy.exp(y0)
            l = y0 - numpy.log(z)
            for i in range(self.n_samples):
                w = self.samples[b, i]
                l += numpy.log(1 - numpy.exp(self.W[w].dot(self.x[b])) / z)

            expect_y[b] = l

        loss = -numpy.sum(expect_y) / self.batch_size
        testing.assert_allclose(y.data, loss, atol=1.e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.W, self.samples)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.W),
            cuda.to_gpu(self.samples))

    def check_backward(self, x_data, t_data, w_data, samples_data):
        gradient_check.check_backward(
            functions.black_out, (x_data, t_data, w_data, samples_data),
            None, atol=1.e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.W, self.samples)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.W),
            cuda.to_gpu(self.samples))


testing.run_module(__name__, __file__)

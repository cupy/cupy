import unittest

import numpy

import chainer
from chainer import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestBlackOut(unittest.TestCase):

    batch_size = 5
    in_size = 4
    count = [3, 2, 1]
    n_samples = 7

    def setUp(self):
        x_shape = (self.batch_size, self.in_size)
        self.x = numpy.random.uniform(
            -1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(
            len(self.count), size=self.batch_size).astype(numpy.int32)

        self.link = links.BlackOut(self.in_size, self.count, self.n_samples)
        self.w = numpy.random.uniform(-1, 1, self.link.W.data.shape)
        self.link.W.data[:] = self.w

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        self.link.sample_data = self.link.sampler.sample(
            (self.batch_size, self.n_samples))
        y = self.link(x, t)

        expect_y = numpy.empty((self.batch_size), dtype=numpy.float32)
        samples = cuda.to_cpu(self.link.sample_data)
        for b in range(self.batch_size):
            z = 0
            for i in range(self.n_samples):
                w = samples[b, i]
                z += numpy.exp(self.w[w].dot(self.x[b]))
            y0 = self.w[self.t[b]].dot(self.x[b])
            z += numpy.exp(y0)
            l = y0 - numpy.log(z)
            for i in range(self.n_samples):
                w = samples[b, i]
                l += numpy.log(1 - numpy.exp(self.w[w].dot(self.x[b])) / z)

            expect_y[b] = l

        loss = -numpy.sum(expect_y) / self.batch_size
        testing.assert_allclose(y.data, loss, atol=1.e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


testing.run_module(__name__, __file__)

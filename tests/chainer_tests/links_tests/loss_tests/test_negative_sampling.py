import unittest

import numpy

import chainer
from chainer import cuda
from chainer.functions.loss import negative_sampling
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    't': [[0, 2], [-1, 1, 2]],
    'reduce': ['sum', 'no'],
}))
class TestNegativeSampling(unittest.TestCase):

    in_size = 3
    sample_size = 2

    def setUp(self):
        batch = len(self.t)
        x_shape = (batch, self.in_size)
        self.link = links.NegativeSampling(
            self.in_size, [10, 5, 2, 5, 2], self.sample_size)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.array(self.t).astype(numpy.int32)

        if self.reduce == 'no':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(numpy.float32)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = self.link(x, t, reduce=self.reduce)
        self.assertEqual(y.shape, self.gy.shape)

        W = cuda.to_cpu(self.link.W.data)
        samples = cuda.to_cpu(y.creator.samples)

        loss = numpy.empty((len(self.x),), numpy.float32)
        for i in range(len(self.x)):
            ix = self.x[i]
            it = self.t[i]
            if it == -1:
                loss[i] = 0
            else:
                w = W[samples[i]]
                f = w.dot(ix)
                # first one is positive example
                f[0] *= -1
                loss[i] = numpy.logaddexp(f, 0).sum()

        if self.reduce == 'sum':
            loss = loss.sum()

        testing.assert_allclose(y.data, loss)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    def test_to_cpu(self):
        self.link.to_gpu()
        self.assertTrue(self.link.sampler.use_gpu)
        self.link.to_cpu()
        self.assertFalse(self.link.sampler.use_gpu)

    @attr.gpu
    def test_backward_cpu_gpu(self):
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        y = self.link(x, t)
        y.backward()

        # fix samples
        negative_sampling.NegativeSamplingFunction.samples = cuda.to_gpu(
           y.creator.samples)
        self.link.to_gpu()
        del negative_sampling.NegativeSamplingFunction.samples
        xg = chainer.Variable(cuda.to_gpu(self.x))
        tg = chainer.Variable(cuda.to_gpu(self.t))
        y_g = self.link(xg, tg)
        y_g.backward()

        testing.assert_allclose(x.grad, xg.grad, atol=1.e-4)


testing.run_module(__name__, __file__)

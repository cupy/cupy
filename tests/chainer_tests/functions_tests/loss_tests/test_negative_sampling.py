import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.loss import negative_sampling
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def make_sampler(xp, high):
    # To fix samples, use fixed samples.
    def sampler(shape):
        return (xp.arange(numpy.prod(shape)) % high).reshape(shape).astype('i')
    return sampler


@testing.parameterize(*testing.product({
    't': [[0, 2], [-1, 1, 2]],
    'reduce': ['sum', 'no'],
}))
class TestNegativeSamplingFunction(unittest.TestCase):

    in_size = 3
    sample_size = 2
    label_size = 5

    def setUp(self):

        batch = len(self.t)
        x_shape = (batch, self.in_size)
        w_shape = (self.label_size, self.in_size)

        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.array(self.t).astype(numpy.int32)
        self.w = numpy.random.uniform(-1, 1, w_shape).astype(numpy.float32)

        if self.reduce == 'no':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(numpy.float32)

    def check_forward(self, x_data, t_data, w_data, sampler):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        w = chainer.Variable(w_data)
        y = functions.negative_sampling(
            x, t, w, sampler, self.sample_size, reduce=self.reduce)
        self.assertEqual(y.shape, self.gy.shape)

        samples = cuda.to_cpu(y.creator.samples)

        loss = numpy.empty((len(self.x),), numpy.float32)
        for i in six.moves.range(len(self.x)):
            ix = self.x[i]
            it = self.t[i]
            if it == -1:
                loss[i] = 0
            else:
                iw = self.w[samples[i]]
                f = iw.dot(ix)
                # first one is positive example
                f[0] *= -1
                loss[i] = numpy.logaddexp(f, 0).sum()

        if self.reduce == 'sum':
            loss = loss.sum()

        testing.assert_allclose(y.data, loss)

    def test_forward_cpu(self):
        self.check_forward(
            self.x, self.t, self.w, make_sampler(numpy, self.label_size))

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.w),
            make_sampler(cuda.cupy, self.label_size))

    def check_backward(self, x_data, t_data, w_data, y_grad, sampler):
        t = chainer.Variable(t_data)

        def f(x, w):
            return negative_sampling.negative_sampling(
                x, t, w, sampler, self.sample_size, self.reduce)

        gradient_check.check_backward(
            f, (x_data, w_data), y_grad, eps=1e-2, atol=1e-4, rtol=1e-4)

    def test_backward_cpu(self):
        self.check_backward(
            self.x, self.t, self.w, self.gy,
            make_sampler(numpy, self.label_size))

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            cuda.to_gpu(self.w), cuda.to_gpu(self.gy),
            make_sampler(cuda.cupy, self.label_size))


class TestNegativeSamplingInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (2,)).astype(numpy.int32)
        self.w = numpy.random.uniform(-1, 1, (5, 3)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        w = xp.asarray(self.w)

        with self.assertRaises(ValueError):
            negative_sampling.negative_sampling(
                x, t, w, make_sampler(xp, 5), 2, reduce='invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

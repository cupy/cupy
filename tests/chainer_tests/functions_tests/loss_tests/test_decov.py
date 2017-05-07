import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _deconv(h):
    h = cuda.to_cpu(h)
    h_mean = h.mean(axis=0)
    N, M = h.shape
    loss_expect = numpy.zeros((M, M), dtype=numpy.float32)
    for i in six.moves.range(M):
        for j in six.moves.range(M):
            if i != j:
                for n in six.moves.range(N):
                    loss_expect[i, j] += (h[n, i] - h_mean[i]) * (
                        h[n, j] - h_mean[j])
    return loss_expect / N


@testing.parameterize(
    {'reduce': 'half_squared_frobenius_norm'},
    {'reduce': 'no'}
)
class TestDeCov(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        if self.reduce == 'half_squared_frobenius_norm':
            gloss_shape = ()
        else:
            gloss_shape = (3, 3)
        self.gloss = numpy.random.uniform(
            -1, 1, gloss_shape).astype(numpy.float32)

    def check_forward(self, h_data):
        h = chainer.Variable(h_data)
        loss = functions.decov(h, self.reduce)
        self.assertEqual(loss.shape, self.gloss.shape)
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        h_data = cuda.to_cpu(h_data)

        loss_expect = _deconv(h_data)
        if self.reduce == 'half_squared_frobenius_norm':
            loss_expect = (loss_expect ** 2).sum() * 0.5

        numpy.testing.assert_allclose(
            loss_expect, loss_value, rtol=1e-4, atol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.h)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.h))

    def check_backward(self, h_data, gloss_data):
        gradient_check.check_backward(
            functions.DeCov(self.reduce), (h_data,), gloss_data,
            eps=0.02, atol=1e-3)

    def check_type(self, h_data, gloss_data):
        h = chainer.Variable(h_data)
        loss = functions.decov(h, self.reduce)
        loss.grad = gloss_data
        loss.backward()
        self.assertEqual(h_data.dtype, h.grad.dtype)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.h, self.gloss)

    def test_backward_type_cpu(self):
        self.check_type(self.h, self.gloss)

    @attr.gpu
    def test_backward_type_gpu(self):
        self.check_type(cuda.to_gpu(self.h),
                        cuda.to_gpu(self.gloss))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.h),
                            cuda.to_gpu(self.gloss))


class TestDeconvInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        h = xp.asarray(self.h)

        with self.assertRaises(ValueError):
            functions.decov(h, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

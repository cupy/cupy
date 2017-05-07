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


def _cross_covariance(y, z):
    row = y.shape[1]
    col = z.shape[1]
    y, z = cuda.to_cpu(y), cuda.to_cpu(z)
    y_mean = y.mean(axis=0)
    z_mean = z.mean(axis=0)
    N = y.shape[0]
    loss_expect = numpy.zeros((row, col), dtype=numpy.float32)
    for i in six.moves.xrange(row):
        for j in six.moves.xrange(col):
            for n in six.moves.xrange(N):
                loss_expect[i, j] += (y[n, i] - y_mean[i]) * (
                    z[n, j] - z_mean[j])
    loss_expect /= N
    return loss_expect


@testing.parameterize(
    {'reduce': 'half_frobenius_norm'},
    {'reduce': 'no'}
)
class TestCrossCovariance(unittest.TestCase):

    def setUp(self):
        self.y = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.z = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)
        if self.reduce == 'half_frobenius_norm':
            gloss_shape = ()
        else:
            gloss_shape = (3, 2)
        self.gloss = numpy.random.uniform(
            -1, 1, gloss_shape).astype(numpy.float32)

    def check_forward(self, y_data, z_data):
        y = chainer.Variable(y_data)
        z = chainer.Variable(z_data)
        loss = functions.cross_covariance(y, z, self.reduce)

        row = y_data.shape[1]
        col = z_data.shape[1]
        self.assertEqual(loss.shape, self.gloss.shape)
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        loss_expect = _cross_covariance(y_data, z_data)
        if self.reduce == 'half_frobenius_norm':
            loss_expect = numpy.sum(loss_expect ** 2) * 0.5
        numpy.testing.assert_allclose(
            loss_expect, loss_value, rtol=1e-4, atol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.y, self.z)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.y), cuda.to_gpu(self.z))

    def check_backward(self, y_data, z_data, gloss_data):
        gradient_check.check_backward(
            functions.CrossCovariance(self.reduce), (y_data, z_data),
            gloss_data, eps=0.02, rtol=1e-4, atol=1e-4)

    def check_type(self, y_data, z_data, gloss_data):
        y = chainer.Variable(y_data)
        z = chainer.Variable(z_data)
        loss = functions.cross_covariance(y, z, self.reduce)
        loss.grad = gloss_data
        loss.backward()
        self.assertEqual(y_data.dtype, y.grad.dtype)
        self.assertEqual(z_data.dtype, z.grad.dtype)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.y, self.z, self.gloss)

    def test_backward_type_cpu(self):
        self.check_type(self.y, self.z, self.gloss)

    @attr.gpu
    def test_backward_type_gpu(self):
        self.check_type(cuda.to_gpu(self.y), cuda.to_gpu(self.z),
                        cuda.to_gpu(self.gloss))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.y), cuda.to_gpu(self.z),
                            cuda.to_gpu(self.gloss))


class TestCrossCovarianceInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.y = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.z = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        y = xp.asarray(self.y)
        z = xp.asarray(self.z)

        with self.assertRaises(ValueError):
            functions.cross_covariance(y, z, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

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


class TestCrossCovariance(unittest.TestCase):

    def setUp(self):
        self.y = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.z = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)

    def check_forward(self, y_data, z_data):
        y = chainer.Variable(y_data)
        z = chainer.Variable(z_data)
        loss = functions.cross_covariance(y, z)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(loss.data)

        # Compute expected value
        y_data, z_data = cuda.to_cpu(y_data), cuda.to_cpu(z_data)
        y_mean = y_data.mean(axis=0)
        z_mean = z_data.mean(axis=0)
        N = y_data.shape[0]

        loss_expect = 0
        for i in six.moves.xrange(y_data.shape[1]):
            for j in six.moves.xrange(z_data.shape[1]):
                ij_loss = 0.
                for n in six.moves.xrange(N):
                    ij_loss += (y_data[n, i] - y_mean[i]) * (
                        z_data[n, j] - z_mean[j])
                ij_loss /= N
                loss_expect += ij_loss ** 2
        loss_expect *= 0.5

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.y, self.z)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.y), cuda.to_gpu(self.z))

    def check_backward(self, y_data, z_data):
        gradient_check.check_backward(
            functions.CrossCovariance(), (y_data, z_data), None, eps=0.02)

    def check_type(self, y_data, z_data):
        y = chainer.Variable(y_data)
        z = chainer.Variable(z_data)
        loss = functions.cross_covariance(y, z)
        loss.backward()
        self.assertEqual(y_data.dtype, y.grad.dtype)
        self.assertEqual(z_data.dtype, z.grad.dtype)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.y, self.z)

    def test_backward_type_cpu(self):
        self.check_type(self.y, self.z)

    @attr.gpu
    def test_backward_type_gpu(self):
        self.check_type(cuda.to_gpu(self.y), cuda.to_gpu(self.z))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.y), cuda.to_gpu(self.z))


testing.run_module(__name__, __file__)

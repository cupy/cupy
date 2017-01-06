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


class TestDeCov(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_forward(self, h_data):
        h = chainer.Variable(h_data)
        loss = functions.decov(h)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(loss.data)

        # Compute expected value
        h_data = cuda.to_cpu(h_data)
        h_mean = h_data.mean(axis=0)
        N = h_data.shape[0]

        loss_expect = 0
        for i in six.moves.range(h_data.shape[1]):
            for j in six.moves.range(h_data.shape[1]):
                ij_loss = 0.
                if i != j:
                    for n in six.moves.range(N):
                        ij_loss += (h_data[n, i] - h_mean[i]) * (
                            h_data[n, j] - h_mean[j])
                    ij_loss /= N
                loss_expect += ij_loss ** 2
        loss_expect *= 0.5

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.h)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.h))

    def check_backward(self, h_data):
        gradient_check.check_backward(
            functions.DeCov(), (h_data,), None, eps=0.02, atol=1e-3)

    def check_type(self, h_data):
        h = chainer.Variable(h_data)
        loss = functions.decov(h)
        loss.backward()
        self.assertEqual(h_data.dtype, h.grad.dtype)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.h)

    def test_backward_type_cpu(self):
        self.check_type(self.h)

    @attr.gpu
    def test_backward_type_gpu(self):
        self.check_type(cuda.to_gpu(self.h))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.h))


testing.run_module(__name__, __file__)

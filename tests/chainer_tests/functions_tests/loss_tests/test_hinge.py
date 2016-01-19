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


class TestHinge(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        # Avoid values around -1.0 for stability
        self.x[numpy.logical_and(-1.01 < self.x, self.x < -0.99)] = 0.5
        self.t = numpy.random.randint(0, 5, (10,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, norm):
        x_val = chainer.Variable(x_data)
        t_val = chainer.Variable(t_data)
        loss = functions.hinge(x_val, t_val, norm)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        for i in six.moves.range(self.x.shape[0]):
            self.x[i, self.t[i]] *= -1
        for i in six.moves.range(self.x.shape[0]):
            for j in six.moves.range(self.x.shape[1]):
                self.x[i, j] = max(0, 1.0 + self.x[i, j])
        loss_expect = 0
        if norm == 'L1':
            loss_expect = numpy.sum(self.x) / self.x.shape[0]
        elif norm == 'L2':
            loss_expect += numpy.sum(self.x ** 2) / self.x.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu_l1(self):
        self.check_forward(self.x, self.t, 'L1')

    @condition.retry(3)
    def test_forward_cpu_l2(self):
        self.check_forward(self.x, self.t, 'L2')

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_l1(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), 'L1')

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_l2(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), 'L2')

    def check_backward(self, x_data, t_data, norm):
        gradient_check.check_backward(
            functions.Hinge(norm), (x_data, t_data), None, eps=0.01, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu_l1(self):
        self.check_backward(self.x, self.t, 'L1')

    @condition.retry(3)
    def test_backward_cpu_l2(self):
        self.check_backward(self.x, self.t, 'L2')

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_l1(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), 'L1')

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_l2(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), 'L2')

testing.run_module(__name__, __file__)

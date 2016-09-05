import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestHuberLoss(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 10)
        self.x = (numpy.random.random(self.shape) - 0.5) * 20
        self.x = self.x.astype(numpy.float32)
        self.t = numpy.random.random(self.shape).astype(numpy.float32)
        self.gy = numpy.random.random(self.shape[0]).astype(numpy.float32)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.huber_loss(x, t, delta=1)
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = cuda.to_cpu(loss.data)

        diff_data = cuda.to_cpu(x_data) - cuda.to_cpu(t_data)
        expected_result = numpy.zeros(self.shape)
        mask = numpy.abs(diff_data) < 1
        expected_result[mask] = 0.5 * diff_data[mask] ** 2
        expected_result[~mask] = numpy.abs(diff_data[~mask]) - 0.5
        loss_expect = numpy.sum(expected_result, axis=1)
        testing.assert_allclose(loss_value, loss_expect)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x_data, t_data, y_grad):
        gradient_check.check_backward(
            functions.HuberLoss(delta=1),
            (x_data, t_data), y_grad, eps=1e-2, atol=1e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

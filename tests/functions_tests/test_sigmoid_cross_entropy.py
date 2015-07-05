import math
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestSigmoidCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (4, 3)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x_val = chainer.Variable(x_data)
        t_val = chainer.Variable(t_data)
        loss = functions.sigmoid_cross_entropy(x_val, t_val, use_cudnn)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0
        for i in six.moves.range(self.x.shape[0]):
            for j in six.moves.range(self.x.shape[1]):
                xd, td = self.x[i, j], self.t[i, j]
                loss_expect -= xd * (td - (xd >= 0)) \
                    - math.log(1 + math.exp(-numpy.abs(xd)))
        loss_expect /= self.t.shape[0]
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.success_at_least(3, 1)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.cudnn
    @condition.success_at_least(3, 1)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.success_at_least(3, 1)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.sigmoid_cross_entropy(x, t, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = gradient_check.numerical_grad(f, (x.data,), (1,), eps=0.01)

        gradient_check.assert_allclose(gx, x.grad)

    @condition.success_at_least(3, 1)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    @condition.success_at_least(3, 1)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.success_at_least(3, 1)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

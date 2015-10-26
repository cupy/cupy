import math
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
        self.x0 = numpy.random.uniform(-1, 1, (5, 2)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (5, 2)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (5,)).astype(numpy.int32)
        self.margin = 1

    def check_forward(self, x0_data, x1_data, t_data, use_cudnn=True):
        x0_val = chainer.Variable(x0_data)
        x1_val = chainer.Variable(x1_data)
        t_val = chainer.Variable(t_data)
        loss = functions.contrastive(x0_val, x1_val, t_val, self.margin,
                                     use_cudnn)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0
        for i in six.moves.range(self.x0.shape[0]):
            x0d, x1d, td = self.x0[i], self.x1[i], self.t[i]
            d = numpy.sum((x0d - x1d) ** 2)
            if td == 1:  # similar pair
                loss_expect += d
            elif td == 0:  # dissimilar pair
                loss_expect += max(1 - math.sqrt(d), 0) ** 2
        loss_expect /= 2.0 * self.t.shape[0]
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                           cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                           cuda.to_gpu(self.t), False)

    def check_backward(self, x0_data, x1_data, t_data, use_cudnn=True):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        t = chainer.Variable(t_data)
        loss = functions.contrastive(x0, x1, t, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x0.data, x1.data, t.data))
        gx0, = gradient_check.numerical_grad(f, (x0.data,), (1,))
        gx1, = gradient_check.numerical_grad(f, (x1.data,), (1,))

        gradient_check.assert_allclose(gx0, x0.grad)
        gradient_check.assert_allclose(gx1, x1.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.t), False)


testing.run_module(__name__, __file__)

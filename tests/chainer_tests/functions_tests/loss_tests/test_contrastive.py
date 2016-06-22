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


@testing.parameterize(
    *testing.product({
        'batchsize': [5, 10], 'input_dim': [2, 3], 'margin': [1, 2]
    })
)
class TestContrastive(unittest.TestCase):

    def setUp(self):
        x_shape = (self.batchsize, self.input_dim)
        self.x0 = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(
            0, 2, (self.batchsize,)).astype(numpy.int32)

    def check_forward(self, x0_data, x1_data, t_data):
        x0_val = chainer.Variable(x0_data)
        x1_val = chainer.Variable(x1_data)
        t_val = chainer.Variable(t_data)
        loss = functions.contrastive(x0_val, x1_val, t_val, self.margin)
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
                loss_expect += max(self.margin - math.sqrt(d), 0) ** 2
        loss_expect /= 2.0 * self.t.shape[0]
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_negative_margin(self):
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward,
                          self.x0, self.x1, self.t)
        self.assertRaises(ValueError, self.check_backward,
                          self.x0, self.x1, self.t)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                           cuda.to_gpu(self.t))

    def check_backward(self, x0_data, x1_data, t_data):
        gradient_check.check_backward(
            functions.Contrastive(self.margin),
            (x0_data, x1_data, t_data), None, rtol=1e-4, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.t))

    @condition.retry(3)
    def test_backward_zero_dist_cpu(self):
        self.check_backward(self.x0, self.x0, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_backward_zero_dist_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x0),
                            cuda.to_gpu(self.t))

testing.run_module(__name__, __file__)

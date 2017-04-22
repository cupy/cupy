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
        'batchsize': [5, 10], 'input_dim': [2, 3], 'margin': [1, 2],
        'reduce': ['mean', 'no']
    })
)
class TestContrastive(unittest.TestCase):

    def setUp(self):
        x_shape = (self.batchsize, self.input_dim)
        self.x0 = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(
            0, 2, (self.batchsize,)).astype(numpy.int32)
        if self.reduce == 'mean':
            self.gy = numpy.float32(numpy.random.uniform(-1, 1))
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (self.batchsize,)).astype(numpy.float32)

    def check_forward(self, x0_data, x1_data, t_data):
        x0_val = chainer.Variable(x0_data)
        x1_val = chainer.Variable(x1_data)
        t_val = chainer.Variable(t_data)
        loss = functions.contrastive(
            x0_val, x1_val, t_val, self.margin, self.reduce)
        self.assertEqual(loss.data.dtype, numpy.float32)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            self.assertEqual(loss.data.shape, (self.batchsize,))
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        loss_expect = numpy.empty((self.batchsize,), numpy.float32)
        for i in six.moves.range(self.x0.shape[0]):
            x0d, x1d, td = self.x0[i], self.x1[i], self.t[i]
            d = numpy.sum((x0d - x1d) ** 2)
            if td == 1:  # similar pair
                loss_expect[i] = d
            elif td == 0:  # dissimilar pair
                loss_expect[i] = max(self.margin - math.sqrt(d), 0) ** 2
            loss_expect[i] /= 2.
        if self.reduce == 'mean':
            loss_expect = numpy.sum(loss_expect) / self.t.shape[0]
        numpy.testing.assert_allclose(loss_expect, loss_value, rtol=1e-5)

    def test_negative_margin(self):
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward,
                          self.x0, self.x1, self.t)
        self.assertRaises(ValueError, self.check_backward,
                          self.x0, self.x1, self.t, self.gy)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                           cuda.to_gpu(self.t))

    def check_backward(self, x0_data, x1_data, t_data, gy_data):
        gradient_check.check_backward(
            functions.Contrastive(self.margin, self.reduce),
            (x0_data, x1_data, t_data), gy_data, rtol=1e-4, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.t, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.t), cuda.to_gpu(self.gy))

    @condition.retry(3)
    def test_backward_zero_dist_cpu(self):
        self.check_backward(self.x0, self.x0, self.t, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_zero_dist_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x0),
                            cuda.to_gpu(self.t), cuda.to_gpu(self.gy))


class TestContrastiveInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (5,)).astype(numpy.int32)

    def check_invalid_option(self, xp):
        x0 = xp.asarray(self.x0)
        x1 = xp.asarray(self.x1)
        t = xp.asarray(self.t)

        with self.assertRaises(ValueError):
            functions.contrastive(x0, x1, t, 1, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

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
        'batchsize': [5, 10], 'input_dim': [2, 3], 'margin': [0.1, 0.5]
    })
)
class TestTriplet(unittest.TestCase):

    def setUp(self):
        x_shape = (self.batchsize, self.input_dim)
        self.a = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.p = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.n = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

    def check_forward(self, a_data, p_data, n_data):
        a_val = chainer.Variable(a_data)
        p_val = chainer.Variable(p_data)
        n_val = chainer.Variable(n_data)
        loss = functions.triplet(a_val, p_val, n_val, self.margin)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        #
        # Compute expected value
        #
        loss_expect = 0
        for i in six.moves.range(self.a.shape[0]):
            ad, pd, nd = self.a[i], self.p[i], self.n[i]
            dp = numpy.sum((ad - pd) ** 2)
            dn = numpy.sum((ad - nd) ** 2)
            loss_expect += max((dp - dn + self.margin), 0)
        loss_expect /= self.a.shape[0]
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_negative_margin(self):
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward,
                          self.a, self.p, self.n)
        self.assertRaises(ValueError, self.check_backward,
                          self.a, self.p, self.n)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.a, self.p, self.n)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.a), cuda.to_gpu(self.p),
                           cuda.to_gpu(self.n))

    def check_backward(self, a_data, p_data, n_data):
        gradient_check.check_backward(
            functions.Triplet(self.margin),
            (a_data, p_data, n_data), None, rtol=1e-4, atol=1e-4)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.a, self.p, self.n)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.p),
                            cuda.to_gpu(self.n))


testing.run_module(__name__, __file__)

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
        'batchsize': [5, 10], 'input_dim': [2, 3],
        'margin': [0.1, 0.5], 'reduce': ['mean', 'no']
    })
)
class TestTriplet(unittest.TestCase):

    def setUp(self):
        x_shape = (self.batchsize, self.input_dim)
        self.a = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.p = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.n = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        if self.reduce == 'mean':
            gy_shape = ()
        else:
            gy_shape = (self.batchsize,)
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(numpy.float32)

    def check_forward(self, a_data, p_data, n_data):
        a_val = chainer.Variable(a_data)
        p_val = chainer.Variable(p_data)
        n_val = chainer.Variable(n_data)
        loss = functions.triplet(a_val, p_val, n_val, self.margin, self.reduce)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            self.assertEqual(loss.data.shape, (self.batchsize,))
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = cuda.to_cpu(loss.data)

        #
        # Compute expected value
        #
        loss_expect = numpy.empty((self.a.shape[0],), dtype=numpy.float32)
        for i in six.moves.range(self.a.shape[0]):
            ad, pd, nd = self.a[i], self.p[i], self.n[i]
            dp = numpy.sum((ad - pd) ** 2)
            dn = numpy.sum((ad - nd) ** 2)
            loss_expect[i] = max((dp - dn + self.margin), 0)
        if self.reduce == 'mean':
            loss_expect = loss_expect.mean()
        numpy.testing.assert_allclose(
            loss_expect, loss_value, rtol=1e-4, atol=1e-4)

    def test_negative_margin(self):
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward,
                          self.a, self.p, self.n)
        self.assertRaises(ValueError, self.check_backward,
                          self.a, self.p, self.n, self.gy)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.a, self.p, self.n)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.a), cuda.to_gpu(self.p),
                           cuda.to_gpu(self.n))

    def check_backward(self, a_data, p_data, n_data, gy_data):
        gradient_check.check_backward(
            functions.Triplet(self.margin, self.reduce),
            (a_data, p_data, n_data), gy_data, rtol=1e-4, atol=1e-4)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.a, self.p, self.n, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.p),
                            cuda.to_gpu(self.n), cuda.to_gpu(self.gy))


class TestContrastiveInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.a = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.p = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.n = numpy.random.randint(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        a = xp.asarray(self.a)
        p = xp.asarray(self.p)
        n = xp.asarray(self.n)

        with self.assertRaises(ValueError):
            functions.triplet(a, p, n, reduce='invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

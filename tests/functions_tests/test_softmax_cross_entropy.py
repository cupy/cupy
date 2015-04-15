from unittest import TestCase

import math
import numpy
import pycuda.gpuarray as gpuarray

from chain import Variable
from chain.gradient_check import numerical_grad, l_infty_dist
from chain.functions import softmax_cross_entropy

class TestSoftmax(TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(-.5, .5, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def test_forward_cpu(self):
        x = Variable(self.x)
        t = Variable(self.t)
        loss = softmax_cross_entropy(x, t)

        loss_expect = self._compute_expected_loss()
        self.assertAlmostEqual(loss_expect, float(loss.data), places=5)

    def test_forward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        t = Variable(gpuarray.to_gpu(self.t))
        loss = softmax_cross_entropy(x, t)

        loss_expect = self._compute_expected_loss()
        self.assertAlmostEqual(loss_expect, float(loss.data.get()), places=5)

    def _compute_expected_loss(self):
        y = numpy.exp(self.x)
        loss_expect = 0
        for i in xrange(y.shape[0]):
            loss_expect -= math.log(y[i, self.t[i]] / y[i].sum())
        return loss_expect

    def test_backward_cpu(self):
        x = Variable(self.x)
        t = Variable(self.t)
        self._check_backward(x, t)

    def test_backward_gpu(self):
        x = Variable(gpuarray.to_gpu(self.x))
        t = Variable(gpuarray.to_gpu(self.t))
        self._check_backward(x, t)

    def _check_backward(self, x, t):
        loss = softmax_cross_entropy(x, t)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = numerical_grad(f, (x.data,), (1,), eps=1e-2)

        self.assertLess(l_infty_dist(gx, x.grad), 1e-4)

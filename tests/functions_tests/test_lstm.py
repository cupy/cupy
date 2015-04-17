from unittest import TestCase

import numpy
import pycuda.gpuarray as gpuarray

from chainer import Variable
from chainer.gradient_check import numerical_grad, l_infty_dist
from chainer.functions import lstm

def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class TestLSTM(TestCase):
    def setUp(self):
        self.c_prev = numpy.random.uniform(-.5, .5, (3, 2, 4)).astype(numpy.float32)
        self.x      = numpy.random.uniform(-1., 1., (3, 8, 4)).astype(numpy.float32)

        self.gc = numpy.random.uniform(-.1, .1, (3, 2, 4)).astype(numpy.float32)
        self.gh = numpy.random.uniform(-.1, .1, (3, 2, 4)).astype(numpy.float32)

    def compute_expected_out(self):
        a_in = self.x[:, [0, 4]]
        i_in = self.x[:, [1, 5]]
        f_in = self.x[:, [2, 6]]
        o_in = self.x[:, [3, 7]]

        c = _sigmoid(i_in) * numpy.tanh(a_in) + _sigmoid(f_in) * self.c_prev
        h = _sigmoid(o_in) * numpy.tanh(c)
        return c, h

    def test_forward_cpu(self):
        c_prev = Variable(self.c_prev)
        x      = Variable(self.x)
        c, h   = lstm(c_prev, x)

        c_expect, h_expect = self.compute_expected_out()

        self.assertLess(l_infty_dist(c_expect, c.data), 1e-8)
        self.assertLess(l_infty_dist(h_expect, h.data), 1e-8)

    def flat(self):
        self.c_prev = self.c_prev[:, :, 0].copy()
        self.x      = self.x[:, :, 0].copy()
        self.gc     = self.gc[:, :, 0].copy()
        self.gh     = self.gh[:, :, 0].copy()

    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()

    def test_forward_gpu(self):
        c_prev = Variable(gpuarray.to_gpu(self.c_prev))
        x      = Variable(gpuarray.to_gpu(self.x))
        c, h   = lstm(c_prev, x)

        c_expect, h_expect = self.compute_expected_out()

        self.assertLess(l_infty_dist(c_expect, c.data.get()), 1e-5)
        self.assertLess(l_infty_dist(h_expect, h.data.get()), 1e-5)

    def test_flat_forward_gpu(self):
        self.flat()
        self.test_forward_gpu()

    def check_backward(self, c_prev, x, c, h):
        func = c.creator
        f = lambda: func.forward((c_prev.data, x.data))
        gc_prev, gx = numerical_grad(f, (c_prev.data, x.data), (c.grad, h.grad))

        self.assertLess(l_infty_dist(gc_prev, c_prev.grad), 1e-5)
        self.assertLess(l_infty_dist(gx,      x.grad),      1e-5)

    def test_full_backward_cpu(self):
        c_prev = Variable(self.c_prev)
        x      = Variable(self.x)
        c, h   = lstm(c_prev, x)
        c.grad = self.gc
        h.grad = self.gh
        c.backward()
        self.check_backward(c_prev, x, c, h)

    def test_flat_full_backward_cpu(self):
        self.flat()
        self.test_full_backward_cpu()

    def test_no_gc_backward_cpu(self):
        c_prev = Variable(self.c_prev)
        x      = Variable(self.x)
        c, h   = lstm(c_prev, x)
        h.grad = self.gh
        h.backward()
        self.check_backward(c_prev, x, c, h)

    def test_flat_no_gc_backward_cpu(self):
        self.flat()
        self.test_no_gc_backward_cpu()

    def test_no_gh_backward_cpu(self):
        c_prev = Variable(self.c_prev)
        x      = Variable(self.x)
        c, h   = lstm(c_prev, x)
        c.grad = self.gc
        c.backward()
        self.check_backward(c_prev, x, c, h)

    def test_flat_no_gh_backward_cpu(self):
        self.flat()
        self.test_no_gh_backward_cpu()

    def test_full_backward_gpu(self):
        c_prev = Variable(gpuarray.to_gpu(self.c_prev))
        x      = Variable(gpuarray.to_gpu(self.x))
        c, h   = lstm(c_prev, x)
        c.grad = gpuarray.to_gpu(self.gc)
        h.grad = gpuarray.to_gpu(self.gh)
        c.backward()
        self.check_backward(c_prev, x, c, h)

    def test_flat_full_backward_gpu(self):
        self.flat()
        self.test_full_backward_gpu()

    def test_no_gc_backward_gpu(self):
        c_prev = Variable(gpuarray.to_gpu(self.c_prev))
        x      = Variable(gpuarray.to_gpu(self.x))
        c, h   = lstm(c_prev, x)
        h.grad = gpuarray.to_gpu(self.gh)
        h.backward()
        self.check_backward(c_prev, x, c, h)

    def test_flat_no_gc_backward_gpu(self):
        self.flat()
        self.test_no_gc_backward_gpu()

    def test_no_gh_backward_gpu(self):
        c_prev = Variable(gpuarray.to_gpu(self.c_prev))
        x      = Variable(gpuarray.to_gpu(self.x))
        c, h   = lstm(c_prev, x)
        c.grad = gpuarray.to_gpu(self.gc)
        c.backward()
        self.check_backward(c_prev, x, c, h)

    def test_flat_no_gh_backward_gpu(self):
        self.flat()
        self.test_no_gh_backward_gpu()

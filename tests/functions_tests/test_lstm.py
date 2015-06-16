from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import lstm
from .. import attr
if cuda.available:
    cuda.init()

def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class TestLSTM(TestCase):
    def setUp(self):
        self.c_prev = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        self.x      = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(numpy.float32)

        self.gc = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        self.gh = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

    def flat(self):
        self.c_prev = self.c_prev[:, :, 0].copy()
        self.x      = self.x[:, :, 0].copy()
        self.gc     = self.gc[:, :, 0].copy()
        self.gh     = self.gh[:, :, 0].copy()

    def check_forward(self, c_prev_data, x_data):
        c_prev = Variable(c_prev_data)
        x      = Variable(x_data)
        c, h   = lstm(c_prev, x)

        # Compute expected out
        a_in = self.x[:, [0, 4]]
        i_in = self.x[:, [1, 5]]
        f_in = self.x[:, [2, 6]]
        o_in = self.x[:, [3, 7]]

        c_expect = _sigmoid(i_in) * numpy.tanh(a_in) + _sigmoid(f_in) * self.c_prev
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)

        assert_allclose(c_expect, c.data)
        assert_allclose(h_expect, h.data)

    def test_forward_cpu(self):
        self.check_forward(self.c_prev, self.x)

    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.c_prev), to_gpu(self.x))

    @attr.gpu
    def test_flat_forward_gpu(self):
        self.flat()
        self.test_forward_gpu()

    def check_backward(self, c_prev_data, x_data, c_grad, h_grad):
        c_prev = Variable(c_prev_data)
        x      = Variable(x_data)
        c, h   = lstm(c_prev, x)
        c.grad = c_grad
        h.grad = h_grad
        c.backward()

        func = c.creator
        f = lambda: func.forward((c_prev.data, x.data))
        gc_prev, gx = numerical_grad(f, (c_prev.data, x.data), (c_grad, h_grad), eps=1e-2)

        assert_allclose(gc_prev, c_prev.grad)
        assert_allclose(gx, x.grad)

    def test_full_backward_cpu(self):
        self.check_backward(self.c_prev, self.x, self.gc, self.gh)

    def test_flat_full_backward_cpu(self):
        self.flat()
        self.test_full_backward_cpu()

    def test_no_gc_backward_cpu(self):
        self.check_backward(self.c_prev, self.x, None, self.gh)

    def test_flat_no_gc_backward_cpu(self):
        self.flat()
        self.test_no_gc_backward_cpu()

    def test_no_gh_backward_cpu(self):
        self.check_backward(self.c_prev, self.x, self.gc, None)

    def test_flat_no_gh_backward_cpu(self):
        self.flat()
        self.test_no_gh_backward_cpu()

    @attr.gpu
    def test_full_backward_gpu(self):
        self.check_backward(
            to_gpu(self.c_prev), to_gpu(self.x), to_gpu(self.gc), to_gpu(self.gh))

    @attr.gpu
    def test_flat_full_backward_gpu(self):
        self.flat()
        self.test_full_backward_gpu()

    @attr.gpu
    def test_no_gc_backward_gpu(self):
        self.check_backward(
            to_gpu(self.c_prev), to_gpu(self.x), None, to_gpu(self.gh))

    @attr.gpu
    def test_flat_no_gc_backward_gpu(self):
        self.flat()
        self.test_no_gc_backward_gpu()

    @attr.gpu
    def test_no_gh_backward_gpu(self):
        self.check_backward(
            to_gpu(self.c_prev), to_gpu(self.x), to_gpu(self.gc), None)

    @attr.gpu
    def test_flat_no_gh_backward_gpu(self):
        self.flat()
        self.test_no_gh_backward_gpu()

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSLSTM(unittest.TestCase):

    def setUp(self):
        self.c_prev1 = numpy.random.uniform(-1,
                                            1, (3, 2, 4)).astype(self.dtype)
        self.c_prev2 = numpy.random.uniform(-1,
                                            1, (3, 2, 4)).astype(self.dtype)
        self.x1 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)
        self.x2 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(self.dtype)

        self.gc = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        self.gh = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def flat(self):
        self.c_prev1 = self.c_prev1[:, :, 0].copy()
        self.c_prev2 = self.c_prev2[:, :, 0].copy()
        self.x1 = self.x1[:, :, 0].copy()
        self.x2 = self.x2[:, :, 0].copy()
        self.gc = self.gc[:, :, 0].copy()
        self.gh = self.gh[:, :, 0].copy()

    def check_forward(self, c_prev1_data, c_prev2_data, x1_data, x2_data):
        c_prev1 = chainer.Variable(c_prev1_data)
        c_prev2 = chainer.Variable(c_prev2_data)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        c, h = functions.slstm(c_prev1, c_prev2, x1, x2)
        self.assertEqual(c.data.dtype, self.dtype)
        self.assertEqual(h.data.dtype, self.dtype)

        # Compute expected out
        a1_in = self.x1[:, [0, 4]]
        i1_in = self.x1[:, [1, 5]]
        f1_in = self.x1[:, [2, 6]]
        o1_in = self.x1[:, [3, 7]]
        a2_in = self.x2[:, [0, 4]]
        i2_in = self.x2[:, [1, 5]]
        f2_in = self.x2[:, [2, 6]]
        o2_in = self.x2[:, [3, 7]]

        c_expect = _sigmoid(i1_in) * numpy.tanh(a1_in) + \
            _sigmoid(i2_in) * numpy.tanh(a2_in) + \
            _sigmoid(f1_in) * self.c_prev1 + \
            _sigmoid(f2_in) * self.c_prev2
        h_expect = _sigmoid(o1_in + o2_in) * numpy.tanh(c_expect)

        testing.assert_allclose(
            c_expect, c.data, **self.check_forward_options)
        testing.assert_allclose(
            h_expect, h.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.c_prev1, self.c_prev2, self.x1, self.x2)

    @condition.retry(3)
    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_prev1),
                           cuda.to_gpu(self.c_prev2),
                           cuda.to_gpu(self.x1),
                           cuda.to_gpu(self.x2))

    @attr.gpu
    @condition.retry(3)
    def test_flat_forward_gpu(self):
        self.flat()
        self.test_forward_gpu()

    def check_backward(self, c_prev1_data, c_prev2_data, x1_data, x2_data,
                       c_grad, h_grad):
        gradient_check.check_backward(
            functions.SLSTM(),
            (c_prev1_data, c_prev2_data, x1_data, x2_data),
            (c_grad, h_grad), **self.check_backward_options)

    @condition.retry(3)
    def test_full_backward_cpu(self):
        self.check_backward(self.c_prev1, self.c_prev2, self.x1, self.x2,
                            self.gc, self.gh)

    @condition.retry(3)
    def test_flat_full_backward_cpu(self):
        self.flat()
        self.test_full_backward_cpu()

    @condition.retry(3)
    def test_no_gc_backward_cpu(self):
        self.check_backward(self.c_prev1, self.c_prev2, self.x1, self.x2,
                            None, self.gh)

    @condition.retry(3)
    def test_flat_no_gc_backward_cpu(self):
        self.flat()
        self.test_no_gc_backward_cpu()

    @condition.retry(3)
    def test_no_gh_backward_cpu(self):
        self.check_backward(self.c_prev1, self.c_prev2, self.x1, self.x2,
                            self.gc, None)

    @condition.retry(3)
    def test_flat_no_gh_backward_cpu(self):
        self.flat()
        self.test_no_gh_backward_cpu()

    @attr.gpu
    @condition.retry(3)
    def test_full_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev1),
            cuda.to_gpu(self.c_prev2),
            cuda.to_gpu(self.x1),
            cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gc),
            cuda.to_gpu(self.gh))

    @attr.gpu
    @condition.retry(3)
    def test_flat_full_backward_gpu(self):
        self.flat()
        self.test_full_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_no_gc_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev1),
            cuda.to_gpu(self.c_prev2),
            cuda.to_gpu(self.x1),
            cuda.to_gpu(self.x2),
            None,
            cuda.to_gpu(self.gh))

    @attr.gpu
    @condition.retry(3)
    def test_flat_no_gc_backward_gpu(self):
        self.flat()
        self.test_no_gc_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_no_gh_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev1),
            cuda.to_gpu(self.c_prev2),
            cuda.to_gpu(self.x1),
            cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gc),
            None)

    @attr.gpu
    @condition.retry(3)
    def test_flat_no_gh_backward_gpu(self):
        self.flat()
        self.test_no_gh_backward_gpu()


testing.run_module(__name__, __file__)

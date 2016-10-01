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


@testing.parameterize(*(testing.product({
    'batch': [3, 2, 0],
    'dtype': [numpy.float32],
}) + testing.product({
    'batch': [3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestLSTM(unittest.TestCase):

    def setUp(self):
        hidden_shape = (3, 2, 4)
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)
        self.c_prev = numpy.random.uniform(
            -1, 1, hidden_shape).astype(self.dtype)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        self.gc = numpy.random.uniform(-1, 1, hidden_shape).astype(self.dtype)
        self.gh = numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def flat(self):
        self.c_prev = self.c_prev[:, :, 0].copy()
        self.x = self.x[:, :, 0].copy()
        self.gc = self.gc[:, :, 0].copy()
        self.gh = self.gh[:, :, 0].copy()

    def check_forward(self, c_prev_data, x_data):
        c_prev = chainer.Variable(c_prev_data)
        x = chainer.Variable(x_data)
        c, h = functions.lstm(c_prev, x)
        self.assertEqual(c.data.dtype, self.dtype)
        self.assertEqual(h.data.dtype, self.dtype)
        batch = len(x_data)

        # Compute expected out
        a_in = self.x[:, [0, 4]]
        i_in = self.x[:, [1, 5]]
        f_in = self.x[:, [2, 6]]
        o_in = self.x[:, [3, 7]]

        c_expect = _sigmoid(i_in) * numpy.tanh(a_in) + \
            _sigmoid(f_in) * self.c_prev[:batch]
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)

        testing.assert_allclose(
            c_expect, c.data[:batch], **self.check_forward_options)
        testing.assert_allclose(
            h_expect, h.data, **self.check_forward_options)
        testing.assert_allclose(
            c_prev_data[batch:], c.data[batch:], **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.c_prev, self.x)

    @condition.retry(3)
    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_flat_forward_gpu(self):
        self.flat()
        self.test_forward_gpu()

    def check_backward(self, c_prev_data, x_data, c_grad, h_grad):
        gradient_check.check_backward(
            functions.LSTM(),
            (c_prev_data, x_data), (c_grad, h_grad),
            **self.check_backward_options)

    @condition.retry(3)
    def test_full_backward_cpu(self):
        self.check_backward(self.c_prev, self.x, self.gc, self.gh)

    @condition.retry(3)
    def test_flat_full_backward_cpu(self):
        self.flat()
        self.test_full_backward_cpu()

    @condition.retry(3)
    def test_no_gc_backward_cpu(self):
        self.check_backward(self.c_prev, self.x, None, self.gh)

    @condition.retry(3)
    def test_flat_no_gc_backward_cpu(self):
        self.flat()
        self.test_no_gc_backward_cpu()

    @condition.retry(3)
    def test_no_gh_backward_cpu(self):
        self.check_backward(self.c_prev, self.x, self.gc, None)

    @condition.retry(3)
    def test_flat_no_gh_backward_cpu(self):
        self.flat()
        self.test_no_gh_backward_cpu()

    @attr.gpu
    @condition.retry(3)
    def test_full_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x),
            cuda.to_gpu(self.gc), cuda.to_gpu(self.gh))

    @attr.gpu
    @condition.retry(3)
    def test_flat_full_backward_gpu(self):
        self.flat()
        self.test_full_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_no_gc_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x),
            None, cuda.to_gpu(self.gh))

    @attr.gpu
    @condition.retry(3)
    def test_flat_no_gc_backward_gpu(self):
        self.flat()
        self.test_no_gc_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_no_gh_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x),
            cuda.to_gpu(self.gc), None)

    @attr.gpu
    @condition.retry(3)
    def test_flat_no_gh_backward_gpu(self):
        self.flat()
        self.test_no_gh_backward_gpu()


testing.run_module(__name__, __file__)

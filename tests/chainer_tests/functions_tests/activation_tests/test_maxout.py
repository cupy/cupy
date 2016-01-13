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


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


def _maxout(x, W, b):
    y = numpy.tensordot(_as_mat(x), W, axes=1)
    if b is not None:
        y += b
    return numpy.max(y, axis=1)


@testing.parameterize(
    {'W_shape': (2, 3, 4), 'b_shape': (3, 4), 'x_shape': (7, 2)},
    {'W_shape': (10, 3, 4), 'b_shape': (3, 4), 'x_shape': (7, 2, 5)},
    {'W_shape': (2, 3, 4), 'b_shape': None, 'x_shape': (7, 2)}
)
class TestNonparameterizedMaxout(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -0.01, 0.01, self.W_shape).astype(numpy.float32)
        for c in six.moves.range(self.W.shape[1]):
            w = numpy.arange(self.W.shape[0], dtype=numpy.float32) + 1
            for o in six.moves.range(self.W.shape[2]):
                self.W[:, c, o] += w * o

        if self.b_shape is not None:
            self.b = numpy.random.uniform(
                -0.01, 0.01, self.b_shape).astype(numpy.float32)
        else:
            self.b = None

        self.x = numpy.random.uniform(
            -0.01, 0.01, self.x_shape).astype(numpy.float32)

        self.y = _maxout(self.x, self.W, self.b)
        self.gy = numpy.random.uniform(
            -0.01, 0.01, self.y.shape).astype(numpy.float32)

    def check_forward(self, x_data, W_data, b_data, y_expect):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        if b_data is None:
            y = functions.maxout(x, W)
        else:
            b = chainer.Variable(b_data)
            y = functions.maxout(x, W, b)
        gradient_check.assert_allclose(y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        b = self.b
        if b is not None:
            b = cuda.to_gpu(b)

        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W),
            b, cuda.to_gpu(self.y))

    def check_backward(self, x_data, W_data, b_data, y_grad):
        if b_data is None:
            gradient_check.check_backward(
                functions.maxout, (x_data, W_data), y_grad,
                eps=1e-2, atol=1e-2)
        else:
            gradient_check.check_backward(
                functions.maxout, (x_data, W_data, b_data), y_grad,
                eps=1e-2, atol=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        b = self.b
        if b is not None:
            b = cuda.to_gpu(b)

        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W),
            b, cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

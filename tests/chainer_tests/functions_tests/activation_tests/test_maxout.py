import unittest

import numpy

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


@testing.parameterize(
    {'W_shape': (2, 3, 4), 'b_shape': (3, 4), 'x_shape': (7, 2)},
    {'W_shape': (10, 3, 4), 'b_shape': (3, 4), 'x_shape': (7, 2, 5)},
    {'W_shape': (2, 3, 4), 'b_shape': None, 'x_shape': (7, 2)}
)
class TestNonparameterizedMaxout(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, self.W_shape).astype(numpy.float32)
        self.x = numpy.random.uniform(
            -1, 1, self.x_shape).astype(numpy.float32)

        self.y = numpy.tensordot(_as_mat(self.x), self.W, axes=1)
        if self.b_shape is not None:
            self.b = numpy.random.uniform(
                -1, 1, self.b_shape).astype(numpy.float32)
            self.y += self.b
        else:
            self.b = None
        self.y = numpy.max(self.y, axis=1)
        self.gy = numpy.random.uniform(
            -1, 1, self.y.shape).astype(numpy.float32)

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
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        if b_data is None:
            y = functions.maxout(x, W)
        else:
            b = chainer.Variable(b_data)
            y = functions.maxout(x, W, b)

        y.grad = y_grad
        y.backward()
        func = y.creator

        if b_data is None:
            f = lambda: func.forward((x.data, W.data))
            gx, gW = gradient_check.numerical_grad(
                f, (x.data, W.data), (y.grad, ), eps=1e-2)
        else:
            f = lambda: func.forward((x.data, W.data, b.data))
            gx, gW, gb = gradient_check.numerical_grad(
                f, (x.data, W.data, b.data), (y.grad, ), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, W.grad)
        if b_data is not None:
            gradient_check.assert_allclose(gb, b.grad)

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

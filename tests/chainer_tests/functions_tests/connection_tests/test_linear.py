import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class TestLinear(unittest.TestCase):

    in_shape = (3,)
    out_size = 2

    def setUp(self):
        in_size = numpy.prod(self.in_shape)
        self.func = functions.Linear(in_size, self.out_size)
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.b = numpy.random.uniform(
            -1, 1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU
        self.b = self.func.b.copy()  # fixed on CPU

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y = self.x.reshape(4, -1).dot(self.func.W.T) + self.func.b

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.func.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, func.W, func.b), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, func.gW)
        gradient_check.assert_allclose(gb, func.gb)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestLinearWithSpatialDimensions(TestLinear):

    in_shape = (3, 2, 2)


class TestInvalidLinear(unittest.TestCase):

    def setUp(self):
        self.func = functions.Linear(3, 2)
        self.x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.func(chainer.Variable(self.x))


class TestNonparameterizedLinear(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, (2, 3)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            -1, 1, 2).astype(numpy.float32)

        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)
        self.y = self.x.dot(self.W.T) + self.b

    def check_forward(self, x_data, W_data, b_data, y_expect):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        if b_data is None:
            y = functions.linear(x, W)
        else:
            b = chainer.Variable(b_data)
            y = functions.linear(x, W, b)
        gradient_check.assert_allclose(y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b,
                           self.x.dot(self.W.T) + self.b)

    @condition.retry(3)
    def test_forward_cpu_nobias(self):
        self.check_forward(self.x, self.W, None, self.x.dot(self.W.T))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b),
            cuda.to_gpu(self.x.dot(self.W.T) + self.b))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_nobias(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), None,
            cuda.to_gpu(self.x.dot(self.W.T)))

    def check_backward(self, x_data, W_data, b_data, y_grad):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = None if b_data is None else chainer.Variable(b_data)
        y = functions.linear(x, W, b)
        y.grad = y_grad
        y.backward()

        func = y.creator
        if b_data is None:
            f = lambda: func.forward((x.data, W.data))
            gx, gW = gradient_check.numerical_grad(
                f, (x.data, W.data), (y.grad,), eps=1e-2)
        else:
            f = lambda: func.forward((x.data, W.data, b.data))
            gx, gW, gb = gradient_check.numerical_grad(
                f, (x.data, W.data, b.data), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, W.grad)
        if b_data is not None:
            gradient_check.assert_allclose(gb, b.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

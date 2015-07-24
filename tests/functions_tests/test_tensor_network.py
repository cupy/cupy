import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


def _check_forward(e1, e2, f, y_expect):
    e1 = chainer.Variable(e1)
    e2 = chainer.Variable(e2)
    y = f(e1, e2)
    gradient_check.assert_allclose(y_expect, y.data)


def _check_backward(e1, e2, y_grad, f, bias):
    e1 = chainer.Variable(e1)
    e2 = chainer.Variable(e2)
    y = f(e1, e2)
    y.grad = y_grad
    y.backward()

    func = y.creator
    f = lambda: func.forward((e1.data, e2.data))

    ge1, ge2, gW = gradient_check.numerical_grad(
        f, (e1.data, e2.data, func.W), (y.grad,), eps=1e-2)

    gradient_check.assert_allclose(ge1, e1.grad)
    gradient_check.assert_allclose(ge2, e2.grad)
    gradient_check.assert_allclose(gW, func.gW)

    if bias:
        gV1, gV2, gb = gradient_check.numerical_grad(
            f, (func.V1, func.V2, func.b),
            (y.grad,), eps=1e-2)
        gradient_check.assert_allclose(gV1, func.gV1)
        gradient_check.assert_allclose(gV2, func.gV2)
        gradient_check.assert_allclose(gb, func.gb)


class TestTensorNetwork(unittest.TestCase):

    in_shape = (2, 3)
    out_size = 4
    batch_size = 10

    def setUp(self):
        self.f = functions.TensorNetwork(
            self.in_shape[0], self.in_shape[1], self.out_size)
        self.f.W = numpy.random.uniform(
            -1, 1, self.f.W.shape).astype(numpy.float32)
        self.f.V1 = numpy.random.uniform(
            -1, 1, self.f.V1.shape).astype(numpy.float32)
        self.f.V2 = numpy.random.uniform(
            -1, 1, self.f.V2.shape).astype(numpy.float32)
        self.f.b = numpy.random.uniform(
            -1, 1, self.f.b.shape).astype(numpy.float32)
        self.f.zero_grads()

        W = self.f.W.copy()
        V1 = self.f.V1.copy()
        V2 = self.f.V2.copy()
        b = self.f.b.copy()

        self.e1 = numpy.random.uniform(
            -1, 1, (self.batch_size, self.in_shape[0])).astype(numpy.float32)
        self.e2 = numpy.random.uniform(
            -1, 1, (self.batch_size, self.in_shape[1])).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (self.batch_size, self.out_size)).astype(numpy.float32)

        self.y = (
            numpy.einsum('ij,ik,jkl->il', self.e1, self.e2, W) +
            self.e1.dot(V1) + self.e2.dot(V2) + b)

    @condition.retry(3)
    def test_forward_cpu(self):
        _check_forward(self.e1, self.e2, self.f, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.f.to_gpu()
        _check_forward(cuda.to_gpu(self.e1),
                       cuda.to_gpu(self.e2),
                       self.f, self.y)

    @condition.retry(3)
    def test_backward_cpu(self):
        _check_backward(self.e1, self.e2, self.gy, self.f, True)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.f.to_gpu()
        _check_backward(cuda.to_gpu(self.e1),
                        cuda.to_gpu(self.e2),
                        cuda.to_gpu(self.gy),
                        self.f, True)


class TestTensorNetworkWOBias(unittest.TestCase):

    in_shape = (2, 3)
    out_size = 4
    batch_size = 10

    def setUp(self):
        self.f = functions.TensorNetwork(
            self.in_shape[0], self.in_shape[1], self.out_size, True)
        self.f.W = numpy.random.uniform(
            -1, 1, self.f.W.shape).astype(numpy.float32)
        self.f.zero_grads()

        W = self.f.W.copy()

        self.e1 = numpy.random.uniform(
            -1, 1, (self.batch_size, self.in_shape[0])).astype(numpy.float32)
        self.e2 = numpy.random.uniform(
            -1, 1, (self.batch_size, self.in_shape[1])).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (self.batch_size, self.out_size)).astype(numpy.float32)

        self.y = numpy.einsum('ij,ik,jkl->il', self.e1, self.e2, W)

    @condition.retry(3)
    def test_forward_cpu(self):
        _check_forward(self.e1, self.e2, self.f, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.f.to_gpu()
        _check_forward(cuda.to_gpu(self.e1),
                       cuda.to_gpu(self.e2),
                       self.f, self.y)

    @condition.retry(3)
    def test_backward_cpu(self):
        _check_backward(self.e1, self.e2, self.gy, self.f, False)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.f.to_gpu()
        _check_backward(cuda.to_gpu(self.e1),
                        cuda.to_gpu(self.e2),
                        cuda.to_gpu(self.gy),
                        self.f, False)

class InitByInitialParameter(unittest.TestCase):

    in_shape = (2, 3)
    out_size = 4
    batch_size = 10

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, (self.in_shape[0], self.in_shape[1], self.out_size)).astype(numpy.float32)
        self.V1 = numpy.random.uniform(
            -1, 1, (self.in_shape[0], self.out_size)).astype(numpy.float32)
        self.V2 = numpy.random.uniform(
            -1, 1, (self.in_shape[1], self.out_size)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            -1, 1, (self.out_size,)).astype(numpy.float32)


class NormalInitialParameter(InitByInitialParameter):

    def check_normal(self, initialW, initial_bias, nobias):
        functions.TensorNetwork(
            self.in_shape[0], self.in_shape[1], self.out_size, nobias,
            initialW, initial_bias)

    def test_normal_cpu_bias(self):
        self.check_normal(self.W, (self.V1, self.V2, self.b), False)

    def test_normal_cpu_nobias(self):
        self.check_normal(self.W, None, False)

    @attr.gpu
    def test_normal_gpu_bias(self):
        initial_bias = (
            cuda.to_gpu(self.V1),
            cuda.to_gpu(self.V2),
            cuda.to_gpu(self.b))
        self.check_normal(cuda.to_gpu(self.W),
                          initial_bias, False)
    
    @attr.gpu
    def test_normal_gpu_nobias(self):
        self.check_normal(cuda.to_gpu(self.W),
                          None, False)


class InvalidInitialParameter(InitByInitialParameter):

    def check_invalid(self, initialW, initial_bias, nobias):
        with self.assertRaises(AssertionError):
            functions.TensorNetwork(
                self.in_shape[0], self.in_shape[1], self.out_size, nobias,
                initialW, initial_bias)

    def test_invalidW_cpu(self):
        W = numpy.random.uniform(
            -1, 1, (self.in_shape[0]+1, self.in_shape[1], self.out_size)).astype(numpy.float32)
        self.check_invalid(W, (self.V1, self.V2, self.b), False)
        self.check_invalid(W, None, True)

    @attr.gpu
    def test_invalidW_gpu(self):
        W = cuda.to_gpu(numpy.random.uniform(
            -1, 1, (self.in_shape[0]+1, self.in_shape[1], self.out_size)).astype(numpy.float32))
        initial_bias = (
            cuda.to_gpu(self.V1),
            cuda.to_gpu(self.V2),
            cuda.to_gpu(self.b))

        self.check_invalid(W, initial_bias, False)
        self.check_invalid(W, None, True)

    def test_invalidV1_cpu(self):
        V1 = numpy.random.uniform(
            -1, 1, (self.in_shape[0]+1, self.out_size)).astype(numpy.float32)
        self.check_invalid(self.W, (V1, self.V2, self.b), False)
    
    @attr.gpu
    def test_invaliV1_gpu(self):
        V1 = numpy.random.uniform(
            -1, 1, (self.in_shape[0]+1, self.out_size)).astype(numpy.float32)
        initial_bias = (
            cuda.to_gpu(V1),
            cuda.to_gpu(self.V2),
            cuda.to_gpu(self.b))

        self.check_invalid(self.W, initial_bias, False)

    def test_invalidV2_cpu(self):
        V2 = numpy.random.uniform(
            -1, 1, (self.in_shape[1]+1, self.out_size)).astype(numpy.float32)
        self.check_invalid(self.W, (self.V1, V2, self.b), False)

    @attr.gpu
    def test_invaliV2_gpu(self):
        V2 = numpy.random.uniform(
            -1, 1, (self.in_shape[1]+1, self.out_size)).astype(numpy.float32)
        initial_bias = (
            cuda.to_gpu(self.V1),
            cuda.to_gpu(V2),
            cuda.to_gpu(self.b))

        self.check_invalid(self.W, initial_bias, False)

    def test_invalidb_cpu(self):
        b = numpy.random.uniform(
            -1, 1, (self.out_size+1,)).astype(numpy.float32)
        self.check_invalid(self.W, (self.V1, self.V2, b), False)

    @attr.gpu
    def test_invalib_gpu(self):
        b = numpy.random.uniform(
            -1, 1, (self.out_size+1,)).astype(numpy.float32)
        initial_bias = (
            cuda.to_gpu(self.V1),
            cuda.to_gpu(self.V2),
            cuda.to_gpu(b))

        self.check_invalid(self.W, initial_bias, False)


testing.run_module(__name__, __file__)

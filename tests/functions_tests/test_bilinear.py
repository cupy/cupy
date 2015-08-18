import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import array


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

    gradient_check.assert_allclose(ge1, e1.grad, rtol=1e-3)
    gradient_check.assert_allclose(ge2, e2.grad, rtol=1e-3)
    gradient_check.assert_allclose(gW, func.gW, rtol=1e-3)

    if bias:
        gV1, gV2, gb = gradient_check.numerical_grad(
            f, (func.V1, func.V2, func.b),
            (y.grad,), eps=1e-2)
        gradient_check.assert_allclose(gV1, func.gV1, rtol=1e-3)
        gradient_check.assert_allclose(gV2, func.gV2, rtol=1e-3)
        gradient_check.assert_allclose(gb, func.gb, rtol=1e-3)


def _batch_to_gpu(*xs):
    return tuple(cuda.to_gpu(x) for x in xs)


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)


class TestBilinear(unittest.TestCase):

    in_shape = (3, 4)
    out_size = 4
    batch_size = 10

    def setUp(self):
        self.f = functions.Bilinear(
            self.in_shape[0], self.in_shape[1], self.out_size)
        self.f.W = _uniform(*self.f.W.shape)
        self.f.V1 = _uniform(*self.f.V1.shape)
        self.f.V2 = _uniform(*self.f.V2.shape)
        self.f.b = _uniform(*self.f.b.shape)
        self.f.zero_grads()

        self.W = self.f.W.copy()
        self.V1 = self.f.V1.copy()
        self.V2 = self.f.V2.copy()
        self.b = self.f.b.copy()

        self.e1 = _uniform(self.batch_size, self.in_shape[0])
        self.e2 = _uniform(self.batch_size, self.in_shape[1])
        self.gy = _uniform(self.batch_size, self.out_size)

        self.y = (
            numpy.einsum('ij,ik,jkl->il', self.e1, self.e2, self.W) +
            self.e1.dot(self.V1) + self.e2.dot(self.V2) + self.b)

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


class TestBilinear2(TestBilinear):

    def setUp(self):
        super(TestBilinear2, self).setUp()

        assert self.in_shape[1] % 2 == 0
        self.e1 = _uniform(self.batch_size, 1, self.in_shape[0])
        self.e2 = _uniform(self.batch_size, self.in_shape[1] // 2, 2)
        self.gy = _uniform(self.batch_size, self.out_size)

        e1 = array.as_mat(self.e1)
        e2 = array.as_mat(self.e2)

        self.y = (
            numpy.einsum('ij,ik,jkl->il', e1, e2, self.W) +
            e1.dot(self.V1) + e2.dot(self.V2) + self.b)


class TestBilinearWOBias(TestBilinear):

    def setUp(self):
        self.f = functions.Bilinear(
            self.in_shape[0], self.in_shape[1], self.out_size, True)
        self.f.W = numpy.random.uniform(
            -1, 1, self.f.W.shape).astype(numpy.float32)
        self.f.zero_grads()

        self.W = self.f.W.copy()

        self.e1 = _uniform(self.batch_size, self.in_shape[0])
        self.e2 = _uniform(self.batch_size, self.in_shape[1])
        self.gy = _uniform(self.batch_size, self.out_size)

        self.y = numpy.einsum('ij,ik,jkl->il', self.e1, self.e2, self.W)

    @condition.retry(3)
    def test_backward_cpu(self):
        _check_backward(self.e1, self.e2, self.gy, self.f, False)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.f.to_gpu()
        _check_backward(cuda.to_gpu(self.e1), cuda.to_gpu(self.e2),
                        cuda.to_gpu(self.gy), self.f, False)


class TestBilinearWOBias2(TestBilinearWOBias):

    def setUp(self):
        super(TestBilinearWOBias2, self).setUp()

        assert self.in_shape[1] % 2 == 0
        self.e1 = _uniform(self.batch_size, 1, self.in_shape[0])
        self.e2 = _uniform(self.batch_size, 2, self.in_shape[1] // 2)
        self.gy = _uniform(self.batch_size, self.out_size)

        e1 = array.as_mat(self.e1)
        e2 = array.as_mat(self.e2)

        self.y = numpy.einsum('ij,ik,jkl->il', e1, e2, self.W)


class InitByInitialParameter(unittest.TestCase):

    in_shape = (2, 3)
    out_size = 4
    batch_size = 10

    def setUp(self):
        self.W = _uniform(self.in_shape[0], self.in_shape[1], self.out_size)
        self.V1 = _uniform(self.in_shape[0], self.out_size)
        self.V2 = _uniform(self.in_shape[1], self.out_size)
        self.b = _uniform(self.out_size,)


class NormalInitialParameter(InitByInitialParameter):

    def check_normal(self, initialW, initial_bias, nobias):
        functions.Bilinear(
            self.in_shape[0], self.in_shape[1], self.out_size, nobias,
            initialW, initial_bias)

    def test_normal_cpu_bias(self):
        self.check_normal(self.W, (self.V1, self.V2, self.b), False)

    def test_normal_cpu_nobias(self):
        self.check_normal(self.W, None, False)

    @attr.gpu
    def test_normal_gpu_bias(self):
        self.check_normal(cuda.to_gpu(self.W),
                          _batch_to_gpu(self.V1, self.V2, self.b), False)

    @attr.gpu
    def test_normal_gpu_nobias(self):
        self.check_normal(cuda.to_gpu(self.W), None, False)


class InvalidInitialParameter(InitByInitialParameter):

    def setUp(self):
        super(InvalidInitialParameter, self).setUp()
        self.invalidW = _uniform(self.in_shape[0] + 1, self.in_shape[1],
                                 self.out_size)
        self.invalidV1 = _uniform(self.in_shape[0] + 1, self.out_size)
        self.invalidV2 = _uniform(self.in_shape[1] + 1, self.out_size)
        self.invalidb = _uniform(self.out_size + 1,)

    def check_invalid(self, initialW, initial_bias, nobias):
        with self.assertRaises(AssertionError):
            functions.Bilinear(
                self.in_shape[0], self.in_shape[1], self.out_size, nobias,
                initialW, initial_bias)

    def test_invalidW_cpu(self):
        self.check_invalid(self.invalidW, (self.V1, self.V2, self.b), False)
        self.check_invalid(self.invalidW, None, True)

    @attr.gpu
    def test_invalidW_gpu(self):
        invalidW = cuda.to_gpu(self.invalidW)
        self.check_invalid(invalidW,
                           _batch_to_gpu(self.V1, self.V2, self.b), False)
        self.check_invalid(invalidW, None, True)

    def test_invalidV1_cpu(self):
        self.check_invalid(self.W, (self.invalidV1, self.V2, self.b), False)

    @attr.gpu
    def test_invaliV1_gpu(self):
        self.check_invalid(cuda.to_gpu(self.W),
                           _batch_to_gpu(self.invalidV1, self.V2, self.b),
                           False)

    def test_invalidV2_cpu(self):
        self.check_invalid(self.W, (self.V1, self.invalidV2, self.b), False)

    @attr.gpu
    def test_invaliV2_gpu(self):
        self.check_invalid(cuda.to_gpu(self.W),
                           _batch_to_gpu(self.V1, self.invalidV2, self.b),
                           False)

    def test_invalidb_cpu(self):
        self.check_invalid(self.W, (self.V1, self.V2, self.invalidb), False)

    @attr.gpu
    def test_invalib_gpu(self):
        self.check_invalid(cuda.to_gpu(self.W),
                           _batch_to_gpu(self.V1, self.V2, self.invalidb),
                           False)


testing.run_module(__name__, __file__)

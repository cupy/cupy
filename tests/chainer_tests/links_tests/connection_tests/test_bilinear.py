import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import array


def _check_forward(e1, e2, f, y_expect):
    e1 = chainer.Variable(e1)
    e2 = chainer.Variable(e2)
    y = f(e1, e2)
    testing.assert_allclose(y_expect, y.data)


def _check_backward(e1, e2, y_grad, link, bias):
    params = [link.W]
    if bias:
        params.append(link.b)

    gradient_check.check_backward(
        link, (e1, e2), y_grad, params, eps=1e-2, rtol=1e-3)


def _batch_to_gpu(*xs):
    return tuple(cuda.to_gpu(x) for x in xs)


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)


class TestBilinear(unittest.TestCase):

    in_shape = (3, 4)
    out_size = 4
    batch_size = 10

    def setUp(self):
        self.f = links.Bilinear(
            self.in_shape[0], self.in_shape[1], self.out_size)
        self.f.W.data[...] = _uniform(*self.f.W.data.shape)
        self.f.V1.data[...] = _uniform(*self.f.V1.data.shape)
        self.f.V2.data[...] = _uniform(*self.f.V2.data.shape)
        self.f.b.data[...] = _uniform(*self.f.b.data.shape)
        self.f.cleargrads()

        self.W = self.f.W.data.copy()
        self.V1 = self.f.V1.data.copy()
        self.V2 = self.f.V2.data.copy()
        self.b = self.f.b.data.copy()

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


class TestBilinear3(TestBilinear):

    out_size = 1


class TestBilinear4(TestBilinear):

    in_shape = (1, 2)


class TestBilinear5(TestBilinear):

    in_shape = (2, 1)


class TestBilinear6(TestBilinear):

    in_shape = (1, 1)


class TestBilinear7(TestBilinear):

    in_shape = (1, 2)
    out_size = 1


class TestBilinear8(TestBilinear):

    in_shape = (2, 1)
    out_size = 1


class TestBilinear9(TestBilinear):

    in_shape = (1, 1)
    out_size = 1


class TestBilinearWOBias(TestBilinear):

    def setUp(self):
        self.f = links.Bilinear(
            self.in_shape[0], self.in_shape[1], self.out_size, True)
        W = self.f.W.data
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        self.f.cleargrads()

        self.W = W.copy()

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


class TestBilinearWOBias3(TestBilinearWOBias):

    out_size = 1


class TestBilinearWOBias4(TestBilinearWOBias):

    in_shape = (1, 2)


class TestBilinearWOBias5(TestBilinearWOBias):

    in_shape = (2, 1)


class TestBilinearWOBias6(TestBilinearWOBias):

    in_shape = (1, 1)


class TestBilinearWOBias7(TestBilinearWOBias):

    in_shape = (1, 2)
    out_size = 1


class TestBilinearWOBias8(TestBilinearWOBias):

    in_shape = (2, 1)
    out_size = 1


class TestBilinearWOBias9(TestBilinearWOBias):

    in_shape = (1, 1)
    out_size = 1


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
        links.Bilinear(
            self.in_shape[0], self.in_shape[1], self.out_size, nobias,
            initialW, initial_bias)

    def test_normal_cpu_bias(self):
        self.check_normal(self.W, (self.V1, self.V2, self.b), False)

    def test_normal_cpu_nobias(self):
        self.check_normal(self.W, None, False)


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
            links.Bilinear(
                self.in_shape[0], self.in_shape[1], self.out_size, nobias,
                initialW, initial_bias)

    def test_invalidW_cpu(self):
        self.check_invalid(self.invalidW, (self.V1, self.V2, self.b), False)

    def test_invalidW_cpu2(self):
        self.check_invalid(self.invalidW, None, True)

    def test_invalidV1_cpu(self):
        self.check_invalid(self.W, (self.invalidV1, self.V2, self.b), False)

    def test_invalidV2_cpu(self):
        self.check_invalid(self.W, (self.V1, self.invalidV2, self.b), False)

    def test_invalidb_cpu(self):
        self.check_invalid(self.W, (self.V1, self.V2, self.invalidb), False)


testing.run_module(__name__, __file__)

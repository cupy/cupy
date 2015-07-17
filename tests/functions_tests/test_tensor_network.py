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


class TestTensorNetwork(unittest.TestCase):

    def setUp(self):
        in_shape = (2, 3)
        out_size = 4

        self.f = functions.TensorNetwork(in_shape, out_size)
        self.f.W = numpy.random.uniform(-1, 1, self.f.W.shape).astype(numpy.float32)
        self.f.V1 = numpy.random.uniform(-1, 1, self.f.V1.shape).astype(numpy.float32)
        self.f.V2 = numpy.random.uniform(-1, 1, self.f.V2.shape).astype(numpy.float32)
        self.f.b = numpy.random.uniform(-1, 1, self.f.b.shape).astype(numpy.float32)
        self.f.zero_grads()

        self.W = self.f.W.copy()
        self.V1 = self.f.V1.copy()
        self.V2 = self.f.V2.copy()
        self.b = self.f.b.copy()

        batch_size = 10
        self.e1 = numpy.random.uniform(-1, 1, (batch_size, in_shape[0])).astype(numpy.float32)
        self.e2 = numpy.random.uniform(-1, 1, (batch_size, in_shape[1])).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (batch_size, out_size)).astype(numpy.float32)

        self.y = (numpy.einsum('ij,ik,jkl->il', self.e1, self.e2, self.W) +
             self.e1.dot(self.V1) +
             self.e2.dot(self.V2) +
             self.b)

    def check_forward(self, x_data):
        e1, e2 = x_data
        e1 = chainer.Variable(e1)
        e2 = chainer.Variable(e2)
        y = self.f(e1, e2)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward((self.e1, self.e2))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.f.to_gpu()
        self.check_forward((cuda.to_gpu(self.e1), cuda.to_gpu(self.e2)))

    def check_backward(self, x_data, y_grad):
        e1, e2 = x_data
        e1 = chainer.Variable(e1)
        e2 = chainer.Variable(e2)
        y = self.f(e1, e2)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((e1.data, e2.data))
        ge1, ge2, gW, gV1, gV2, gb = gradient_check.numerical_grad(
            f, (e1.data, e2.data, func.W, func.V1, func.V2, func.b), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(ge1, e1.grad)
        gradient_check.assert_allclose(ge2, e2.grad)
        gradient_check.assert_allclose(gW, func.gW)
        gradient_check.assert_allclose(gV1, func.gV1)
        gradient_check.assert_allclose(gV2, func.gV2)
        gradient_check.assert_allclose(gb, func.gb)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward((self.e1, self.e2), self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.f.to_gpu()
        self.check_backward(
            (cuda.to_gpu(self.e1), cuda.to_gpu(self.e2)),
            cuda.to_gpu(self.gy))



testing.run_module(__name__, __file__)

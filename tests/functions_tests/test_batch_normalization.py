from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import BatchNormalization
from .. import attr
if cuda.available:
    cuda.init()

# fully-connected usage
class TestBatchNormalization(TestCase):
    aggr_axes = 0

    def setUp(self):
        self.func = BatchNormalization(3)
        self.func.gamma = numpy.random.uniform(
            .5, 1, self.func.gamma.shape).astype(numpy.float32)
        self.func.beta = numpy.random.uniform(
            -1, 1, self.func.beta.shape).astype(numpy.float32)
        self.func.ggamma.fill(0)
        self.func.gbeta.fill(0)

        self.gamma = self.func.gamma.copy().reshape(1, 3)  # fixed on CPU
        self.beta  = self.func.beta.copy().reshape(1, 3)   # fixed on CPU

        self.x  = numpy.random.uniform(-1, 1, (7, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (7, 3)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = self.func(x)

        mean = self.x.mean(axis=self.aggr_axes, keepdims=True)
        std  = numpy.sqrt(self.x.var(axis=self.aggr_axes, keepdims=True) + self.func.eps)
        y_expect = self.gamma * (self.x - mean) / std + self.beta

        assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.func.to_gpu()
        self.check_forward(to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, ggamma, gbeta = numerical_grad(f, (x.data, func.gamma, func.beta), (y.grad,), eps=1e-2)

        assert_allclose(gx, x.grad, rtol=1e-3, atol=1e-4)
        assert_allclose(ggamma, func.ggamma)
        assert_allclose(gbeta, func.gbeta)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))


# convolutional usage
class TestBatchNormalization2D(TestBatchNormalization):
    aggr_axes = 0, 2, 3

    def setUp(self):
        self.func = BatchNormalization(3)
        self.func.gamma = numpy.random.uniform(
            .5, 1, self.func.gamma.shape).astype(numpy.float32)
        self.func.beta = numpy.random.uniform(
            -1, 1, self.func.beta.shape).astype(numpy.float32)
        self.func.ggamma.fill(0)
        self.func.gbeta.fill(0)

        self.gamma = self.func.gamma.copy().reshape(1, 3, 1, 1)  # fixed on CPU
        self.beta  = self.func.beta.copy().reshape(1, 3, 1, 1)   # fixed on CPU

        self.x  = numpy.random.uniform(-1, 1, (7, 3, 2, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (7, 3, 2, 2)).astype(numpy.float32)

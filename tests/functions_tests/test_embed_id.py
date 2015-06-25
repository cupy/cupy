from unittest import TestCase
import numpy
from six.moves import range
from chainer import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import EmbedID
from chainer.testing import attr

if cuda.available:
    cuda.init()


class TestEmbedID(TestCase):

    def setUp(self):
        self.func = EmbedID(3, 2)
        self.func.gW.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU
        self.x = numpy.array([0, 1, 0], dtype=numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def to_gpu(self):
        self.func.W = to_gpu(self.func.W)
        self.func.gW = to_gpu(self.func.gW)

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = self.func(x)

        y_expect = numpy.empty_like(self.gy)
        for i in range(self.x.size):
            y_expect[i] = self.W[int(self.x[i])]

        assert_allclose(y_expect, y.data, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.to_gpu()
        self.check_forward(to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gW, = numerical_grad(f, (func.W,), (y.grad,))
        assert_allclose(gW, func.gW)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.to_gpu()
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

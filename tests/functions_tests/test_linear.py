from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import Linear
from .. import attr

if cuda.available:
    cuda.init()

class TestLinear(TestCase):
    def setUp(self):
        self.func = Linear(3, 2)
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.b = numpy.random.uniform(
            -1, 1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)

        self.W  = self.func.W.copy()  # fixed on CPU
        self.b  = self.func.b.copy()  # fixed on CPU

        self.x  = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)
        self.y  = self.x.dot(self.func.W.T) + self.func.b

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = self.func(x)
        y_expect = self.x.dot(self.W.T) + self.b
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
        gx, gW, gb = numerical_grad(f, (x.data, func.W, func.b), (y.grad,), eps=1e-2)

        assert_allclose(gx, x.grad)
        assert_allclose(gW, func.gW)
        assert_allclose(gb, func.gb)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

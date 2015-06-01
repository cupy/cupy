from unittest import TestCase
import numpy
from chainer import cuda, Variable
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import PReLU

cuda.init()

class TestPReLUSingle(TestCase):
    def setUp(self):
        self.func = PReLU()
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.gW.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU

        # Avoid unstability of numerical gradient
        self.x  = numpy.random.uniform(.5, 1, (4, 3, 2)).astype(numpy.float32)
        self.x *= numpy.random.randint(2, size=(4, 3, 2)) * 2 - 1
        self.gy = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = self.func(x)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                y_expect[i] *= self.W

        assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_gpu(self):
        self.func.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW = numerical_grad(f, (x.data, func.W), (y.grad,))

        assert_allclose(gx, x.grad)
        assert_allclose(gW, func.gW, atol=1e-4)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestPReLUMulti(TestPReLUSingle):
    def setUp(self):
        self.func = PReLU(shape=(3,))
        self.func.W = numpy.random.uniform(
            -1, 1, self.func.W.shape).astype(numpy.float32)
        self.func.gW.fill(0)

        self.W = self.func.W.copy()  # fixed on CPU

        # Avoid unstability of numerical gradient
        self.x  = numpy.random.uniform(.5, 1, (4, 3, 2)).astype(numpy.float32)
        self.x *= numpy.random.randint(2, size=(4, 3, 2)) * 2 - 1
        self.gy = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = self.func(x)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                y_expect[i] *= self.W[i[1]]

        assert_allclose(y_expect, y.data)

from unittest import TestCase
import random
import numpy
from chainer import cuda, Variable
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import leaky_relu

cuda.init()

class TestLeakyReLU(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-1, 1, (5, 4)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (5, 4)).astype(numpy.float32)
        self.slope = random.random()

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = leaky_relu(x, slope=self.slope)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] *= self.slope

        assert_allclose(expected, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = leaky_relu(x, slope=self.slope)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

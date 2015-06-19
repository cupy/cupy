from unittest import TestCase
import numpy
from chainer import cuda, Variable
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import local_response_normalization
from chainer.testing import attr

if cuda.available:
    cuda.init()

class TestLocalResponseNormalization(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(-1, 1, (2, 7, 3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 7, 3, 2)).astype(numpy.float32)

    def check_forward(self, x_data):
        x = Variable(x_data)
        y = local_response_normalization(x)
        y_data = cuda.to_cpu(y.data)

        # Naive implementation
        y_expect = numpy.zeros_like(self.x)
        for n, c, h, w in numpy.ndindex(self.x.shape):
            s = 0
            for i in xrange(max(0, c - 2), min(7, c + 2)):
                s += self.x[n, i, h, w] ** 2
            denom = (2 + 1e-4 * s) ** .75
            y_expect[n, c, h, w] = self.x[n, c, h, w] / denom

        assert_allclose(y_expect, y_data, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = Variable(x_data)
        y = local_response_normalization(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,), eps=1)

        assert_allclose(gx, x.grad, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

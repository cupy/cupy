from unittest import TestCase

import numpy

from chainer import cuda
from chainer.cuda import to_gpu
from chainer.functions import relu
from chainer.gradient_check import assert_allclose
from chainer.gradient_check import numerical_grad
from chainer.testing import attr
from chainer import Variable


if cuda.available:
    cuda.init()


class TestReLU(TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.x *= numpy.random.randint(2, size=(3, 2)) * 2 - 1
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        x = Variable(x_data)
        y = relu(x, use_cudnn=use_cudnn)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_backward_cpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy), False)

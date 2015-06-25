from unittest import TestCase

import numpy
from six.moves import range

from chainer import cuda
from chainer.cuda import to_gpu
from chainer.functions import softmax
from chainer.gradient_check import assert_allclose
from chainer.gradient_check import numerical_grad
from chainer.testing import attr
from chainer import Variable


if cuda.available:
    cuda.init()


class TestSoftmax(TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = Variable(x_data)
        y = softmax(x, use_cudnn)

        y_expect = numpy.exp(self.x)
        for i in range(y_expect.shape[0]):
            y_expect[i] /= y_expect[i].sum()

        assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x))

    @attr.gpu
    def test_forwrad_gpu_no_cudnn(self):
        self.check_forward(to_gpu(self.x), False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        x = Variable(x_data)
        y = softmax(x, use_cudnn)
        y.grad = gy_data
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,), eps=1e-2)

        assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy), False)

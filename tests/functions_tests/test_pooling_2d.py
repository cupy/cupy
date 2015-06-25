from unittest import TestCase

import numpy
from six.moves import range

from chainer import cuda
from chainer.cuda import to_cpu
from chainer.cuda import to_gpu
from chainer.functions import average_pooling_2d
from chainer.functions import max_pooling_2d
from chainer.gradient_check import assert_allclose
from chainer.gradient_check import numerical_grad
from chainer.testing import attr
from chainer import Variable


if cuda.available:
    cuda.init()


class TestMaxPooling2D(TestCase):
    cover_all = False

    def setUp(self):
        # Avoid unstability of numerical gradient
        self.x = numpy.arange(
            2 * 3 * 4 * 3, dtype=numpy.float32).reshape(2, 3, 4, 3)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1

        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 3, 2, 2)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = Variable(x_data)
        y = max_pooling_2d(x, 3, stride=2, pad=1, cover_all=self.cover_all,
                           use_cudnn=use_cudnn)
        y_data = to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in range(2):
            for c in range(3):
                if self.cover_all:
                    expect = numpy.array([
                        [self.x[k, c, 0:2, 0:2].max(), self.x[
                            k, c, 0:2, 1:3].max()],
                        [self.x[k, c, 1:4, 0:2].max(), self.x[
                            k, c, 1:4, 1:3].max()],
                        [self.x[k, c, 3:4, 0:2].max(), self.x[
                            k, c, 3:4, 1:3].max()]])
                else:
                    expect = numpy.array([
                        [self.x[k, c, 0:2, 0:2].max(), self.x[
                            k, c, 0:2, 1:3].max()],
                        [self.x[k, c, 1:4, 0:2].max(), self.x[
                            k, c, 1:4, 1:3].max()]])
                assert_allclose(expect, y_data[k, c])

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(to_gpu(self.x), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        x = Variable(x_data)
        y = max_pooling_2d(x, 3, stride=2, pad=1, cover_all=self.cover_all,
                           use_cudnn=use_cudnn)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(to_cpu(gx), to_cpu(x.grad))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy), False)


class TestMaxPooling2DCoverAll(TestMaxPooling2D):
    cover_all = True

    def setUp(self):
        super(TestMaxPooling2DCoverAll, self).setUp()
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 3, 3, 2)).astype(numpy.float32)


class TestAveragePooling2D(TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 3, 2, 2)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = Variable(x_data)
        y = average_pooling_2d(x, 3, stride=2, pad=1, use_cudnn=use_cudnn)
        y_data = to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in range(2):
            for c in range(3):
                expect = numpy.array([
                    [self.x[k, c, 0:2, 0:2].sum(), self.x[
                        k, c, 0:2, 1:3].sum()],
                    [self.x[k, c, 1:4, 0:2].sum(), self.x[
                        k, c, 1:4, 1:3].sum()]]) / 9
                assert_allclose(expect, y_data[k, c])

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(to_gpu(self.x), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        x = Variable(x_data)
        y = average_pooling_2d(x, 3, stride=2, pad=1, use_cudnn=use_cudnn)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,), eps=1e-2)

        assert_allclose(to_cpu(gx), to_cpu(x.grad))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(to_gpu(self.x), to_gpu(self.gy), False)

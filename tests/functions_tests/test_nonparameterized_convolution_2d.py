import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestNonparameterizedConvolution2D(unittest.TestCase):

    def setUp(self, use_cudnn=True):
        in_channels = 3
        out_channels = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 1
        self.use_cudnn = use_cudnn
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            -1, 1, out_channels).astype(numpy.float32)

        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 2, 2, 2)).astype(numpy.float32)

    @attr.cudnn
    def test_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = chainer.Variable(self.b)
        y_cpu = functions.convolution_2d(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = functions.convolution_2d(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.use_cudnn = False
        self.test_forward_consistency()

    def check_backward(self, x_data, W_data, b_data, y_grad):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)

        y = functions.convolution_2d(x, W, b, stride=self.stride, pad=self.pad,
                                     use_cudnn=self.use_cudnn)

        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, W.data, b.data))
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, W.data, b.data), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, W.grad)
        gradient_check.assert_allclose(gb, b.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_im2col(self):
        self.use_cudnn = False
        self.test_backward_gpu()

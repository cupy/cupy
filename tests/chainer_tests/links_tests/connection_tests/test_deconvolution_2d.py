import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links as L
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.testing import parameterize
from chainer.utils import conv


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


@parameterize(
    *testing.product({
        'nobias': [True, False],
        'use_cudnn': [True, False]
    })
)
class TestDeconvolution2D(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ksize = 3
        stride = 2
        pad = 1
        self.link = L.Deconvolution2D(
            in_channels, out_channels, ksize,
            stride=stride, pad=pad, nobias=self.nobias)
        self.link.W.data[...] = numpy.random.uniform(
            -1, 1, self.link.W.data.shape).astype(numpy.float32)
        if not self.nobias:
            self.link.b.data[...] = numpy.random.uniform(
                -1, 1, self.link.b.data.shape).astype(numpy.float32)

        self.link.cleargrads()

        N = 2
        h, w = 3, 2
        kh, kw = _pair(ksize)
        out_h = conv.get_deconv_outsize(h, kh, stride, pad)
        out_w = conv.get_deconv_outsize(w, kw, stride, pad)
        self.gy = numpy.random.uniform(
            -1, 1, (N, out_channels, out_h, out_w)).astype(numpy.float32)
        self.x = numpy.random.uniform(
            -1, 1, (N, in_channels, h, w)).astype(numpy.float32)

    def check_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)

        self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)

        testing.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency(self):
        self.link.use_cudnn = self.use_cudnn
        self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        params = [self.link.W]
        if not self.nobias:
            params.append(self.link.b)

        gradient_check.check_backward(
            self.link, x_data, y_grad, params, eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.use_cudnn = self.use_cudnn
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@parameterize(
    *testing.product({
        'nobias': [True, False],
        'use_cudnn': [True, False]
    })
)
class TestDeconvolution2DParameterShapePlaceholder(unittest.TestCase):

    def setUp(self):
        out_channels = 2
        ksize = 3
        stride = 2
        pad = 1
        in_channels = None
        self.link = L.Deconvolution2D(
            in_channels, out_channels, ksize,
            stride=stride, pad=pad, nobias=self.nobias)
        if not self.nobias:
            self.link.b.data[...] = numpy.random.uniform(
                -1, 1, self.link.b.data.shape).astype(numpy.float32)

        N = 2
        h, w = 3, 2
        kh, kw = _pair(ksize)
        out_h = conv.get_deconv_outsize(h, kh, stride, pad)
        out_w = conv.get_deconv_outsize(w, kw, stride, pad)
        self.gy = numpy.random.uniform(
            -1, 1, (N, out_channels, out_h, out_w)).astype(numpy.float32)
        self.x = numpy.random.uniform(
            -1, 1, (N, 3, h, w)).astype(numpy.float32)
        self.link(chainer.Variable(self.x))
        self.link.cleargrads()

    def check_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)

        self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)

        testing.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency(self):
        self.link.use_cudnn = self.use_cudnn
        self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        params = [self.link.W]
        if not self.nobias:
            params.append(self.link.b)

        gradient_check.check_backward(
            self.link, x_data, y_grad, params, eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.use_cudnn = self.use_cudnn
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

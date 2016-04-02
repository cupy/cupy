import unittest

import mock
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.connection import deconvolution_2d
from chainer import gradient_check
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
        'in_channels': [3],
        'out_channels': [2],
        'wscale': [1],
        'ksize': [3],
        'stride': [1, 2],
        'pad': [1],
        'nobias': [True, False],
        'use_cudnn': [True, False],
        'test_outsize': [True, False],
        'c_contiguous': [True, False],
    })
)
class TestDeconvolution2DFunction(unittest.TestCase):

    def setUp(self, use_cudnn=True):
        kh, kw = _pair(self.ksize)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.pad)
        self.W = numpy.random.normal(
            0, self.wscale * numpy.sqrt(1. / (kh * kw * self.in_channels)),
            (self.in_channels, self.out_channels, kh, kw)
        ).astype(numpy.float32)
        self.b = None if self.nobias else numpy.random.uniform(
            -1, 1, self.out_channels).astype(numpy.float32)

        N = 2
        inh, inw = 4, 3
        outh = conv.get_deconv_outsize(inh, kh, sh, ph)
        outw = conv.get_deconv_outsize(inw, kw, sw, pw)
        self.outsize = (outh, outw) if self.test_outsize else None
        self.x = numpy.random.uniform(
            -1, 1, (N, self.in_channels, inh, inw)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (N, self.out_channels, outh, outw)).astype(numpy.float32)

    @attr.cudnn
    def test_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if self.nobias else chainer.Variable(self.b)
        y_cpu = F.deconvolution_2d(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            outsize=self.outsize, use_cudnn=self.use_cudnn)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if self.nobias else chainer.Variable(
            cuda.to_gpu(self.b))
        y_gpu = F.deconvolution_2d(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            outsize=self.outsize, use_cudnn=self.use_cudnn)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.test_forward_consistency()

    def check_backward(self, x_data, W_data, b_data, y_grad):
        xp = cuda.get_array_module(x_data)
        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            deconvolution_2d.Deconvolution2DFunction(
                self.stride, self.pad, self.outsize, self.use_cudnn),
            args, y_grad, eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        b = None if self.b is None else cuda.to_gpu(self.b)
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            b, cuda.to_gpu(self.gy))


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
@attr.cudnn
class TestDeconvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.in_channels = 3
        self.out_channels = 2
        kh, kw = _pair(3)
        sh, sw = _pair(1)
        ph, pw = _pair(1)
        self.W = cuda.cupy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * self.in_channels)),
            (self.in_channels, self.out_channels, kh, kw)
        ).astype(numpy.float32)
        N = 2
        inh, inw = 4, 3
        outh = conv.get_deconv_outsize(inh, kh, sh, ph)
        outw = conv.get_deconv_outsize(inw, kw, sw, pw)
        self.x = cuda.cupy.random.uniform(
            -1, 1, (N, self.in_channels, inh, inw)).astype(numpy.float32)
        self.gy = cuda.cupy.random.uniform(
            -1, 1, (N, self.out_channels, outh, outw)).astype(numpy.float32)

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return F.deconvolution_2d(
            x, W, None, stride=1, pad=1, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        v2 = 'cupy.cudnn.cudnn.convolutionBackwardData_v2'
        v3 = 'cupy.cudnn.cudnn.convolutionBackwardData_v3'
        with mock.patch(v2) as func_v2, mock.patch(v3) as func_v3:
            self.forward()
            self.assertEqual(func_v2.called or func_v3.called, self.use_cudnn)

    def test_call_cudnn_backrward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.convolutionForward') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn)


testing.run_module(__name__, __file__)

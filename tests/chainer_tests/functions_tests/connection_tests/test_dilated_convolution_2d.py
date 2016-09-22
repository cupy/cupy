import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import dilated_convolution_2d
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'c_contiguous': [True, False],
    'cover_all': [True, False],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}) + testing.product({
    'c_contiguous': [False],
    'cover_all': [False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestDilatedConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 2
        self.dilate = 2
        self.use_cudnn = True
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, out_channels).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.x_dtype)
        if self.cover_all:
            self.gy = numpy.random.uniform(-1, 1,
                                           (2, 2, 3, 2)).astype(self.x_dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 2, 2, 2)).astype(self.x_dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    @attr.gpu
    def test_forward_consistency(self, nobias=False):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = functions.dilated_convolution_2d(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            dilate=self.dilate, use_cudnn=self.use_cudnn,
            cover_all=self.cover_all)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = functions.dilated_convolution_2d(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            dilate=self.dilate, use_cudnn=self.use_cudnn,
            cover_all=self.cover_all)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data.get(), **self.check_forward_options)

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.use_cudnn = False
        self.test_forward_consistency()

    @attr.gpu
    def test_forward_consistency_im2col_nobias(self):
        self.use_cudnn = False
        self.test_forward_consistency(nobias=True)

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
            dilated_convolution_2d.DilatedConvolution2DFunction(
                self.stride, self.pad, self.dilate,
                self.use_cudnn, self.cover_all),
            args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.use_cudnn = False
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col_nobias(self):
        self.use_cudnn = False
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestDilatedConvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 2
        self.dilate = 2
        self.x = cuda.cupy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.dtype)
        self.W = cuda.cupy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(
            -1, 1, (2, 2, 2, 2)).astype(self.dtype)
        self.expect = self.use_cudnn and (
            cuda.cudnn.cudnn.getVersion() >= 3000 or
            self.dtype != numpy.float16)

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return functions.dilated_convolution_2d(
            x, W, None, stride=self.stride, pad=self.pad, dilate=self.dilate,
            use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.convolutionForward') as func:
            self.forward()
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backrward(self):
        y = self.forward()
        y.grad = self.gy
        if cuda.cudnn.cudnn.getVersion() >= 4000:
            name = 'cupy.cudnn.cudnn.convolutionBackwardData_v3'
        else:
            name = 'cupy.cudnn.cudnn.convolutionBackwardData_v2'
        with mock.patch(name) as func:
            y.backward()
            self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)

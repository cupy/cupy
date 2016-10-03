import unittest

import functools
import mock
import numpy
from operator import mul

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.connection import deconvolution_nd
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.testing import parameterize
from chainer.utils import conv
from chainer.utils import type_check


@parameterize(*testing.product({
    'dims': [(5, 4, 3), (4, 3), (3,)],
    'nobias': [False],
    'test_outsize': [False],
    'c_contiguous': [True],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}) + testing.product({
    'dims': [(5, 4, 3)],
    'nobias': [True, False],
    'test_outsize': [True, False],
    'c_contiguous': [True, False],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
class TestDeconvolutionND(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim

        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (in_channels, out_channels) + ksize
        self.W = numpy.random.normal(0, W_scale, W_shape).astype(self.W_dtype)
        self.b = numpy.random.uniform(-1, 1, out_channels).astype(self.x_dtype)

        outs = tuple(
            conv.get_deconv_outsize(d, k, s, p)
            for (d, k, s, p) in zip(self.dims, ksize, self.stride, self.pad))
        self.outsize = outs if self.test_outsize else None
        x_shape = (2, in_channels) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        gy_shape = (2, out_channels) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.x_dtype)

        self.test_forward_options = {}
        self.check_backward_options = {
            'eps': 1e-2, 'atol': 1e-4, 'rtol': 1e-3}
        if self.x_dtype == numpy.float16:
            self.test_forward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_backward_options = {
                'eps': 2 ** -3, 'atol': 1e-2, 'rtol': 1e-1}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {
                'eps': 2 ** -3, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward_consistency(self, use_cudnn=True):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if self.nobias else chainer.Variable(self.b)
        y_cpu = F.deconvolution_nd(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            outsize=self.outsize, use_cudnn=use_cudnn)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if self.nobias else chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = F.deconvolution_nd(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            outsize=self.outsize, use_cudnn=use_cudnn)

        self.assertEqual(y_cpu.data.dtype, self.x_dtype)
        self.assertEqual(y_gpu.data.dtype, self.x_dtype)
        testing.assert_allclose(
            y_cpu.data, y_gpu.data.get(), **self.test_forward_options)

    @attr.cudnn
    def test_forward_consistency_cudnn(self):
        self.check_forward_consistency(use_cudnn=True)

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.check_forward_consistency(use_cudnn=False)

    def check_forward_consistency_regression(self, x_data, W_data, b_data,
                                             use_cudnn=True):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = None if self.nobias else chainer.Variable(b_data)

        y_nd = F.deconvolution_nd(x, W, b, stride=self.stride, pad=self.pad,
                                  outsize=self.outsize, use_cudnn=use_cudnn)
        y_2d = F.deconvolution_2d(x, W, b, stride=self.stride, pad=self.pad,
                                  outsize=self.outsize, use_cudnn=use_cudnn)

        testing.assert_allclose(
            y_nd.data, y_2d.data, **self.test_forward_options)

    def test_forward_consistency_regression_cpu(self):
        # Regression test to deconvolution_nd.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(self.x, self.W, self.b)

    @attr.cudnn
    def test_forward_consistency_regression_cudnn(self):
        # Regression test to deconvolution_nd.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(
                cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b),
                use_cudnn=True)

    @attr.gpu
    def test_forward_consistency_regression_im2col(self):
        # Regression test to deconvolution_nd.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(
                cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b),
                use_cudnn=False)

    def check_backward(self, x_data, W_data, b_data, y_grad, use_cudnn=False):
        if not self.c_contiguous:
            xp = cuda.get_array_module(x_data)
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

        inputs = (x_data, W_data)
        if b_data is not None:
            inputs = inputs + (b_data,)

        ndim = len(self.dims)
        gradient_check.check_backward(
            deconvolution_nd.DeconvolutionND(
                ndim, self.stride, self.pad, self.outsize, use_cudnn),
            inputs, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_cudnn(self):
        b = None if self.b is None else cuda.to_gpu(self.b)
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), b,
            cuda.to_gpu(self.gy), use_cudnn=True)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        b = None if self.b is None else cuda.to_gpu(self.b)
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), b,
            cuda.to_gpu(self.gy), use_cudnn=False)


@testing.parameterize(*testing.product({
    'dims': [(5, 4, 3), (4, 3), (3,)],
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestDeconvolutionNDCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        stride = (1,) * ndim
        pad = (1,) * ndim

        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (in_channels, out_channels) + ksize
        self.W = cuda.cupy.random.normal(
            0, W_scale, W_shape).astype(self.dtype)
        outs = tuple(
            conv.get_deconv_outsize(d, k, s, p)
            for (d, k, s, p) in zip(self.dims, ksize, stride, pad))
        x_shape = (2, in_channels) + self.dims
        self.x = cuda.cupy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        gy_shape = (2, out_channels) + outs
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        self.expected = self.use_cudnn and ndim > 1 and (
            cuda.cudnn.cudnn.getVersion() >= 3000 or
            self.dtype != numpy.float16)

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return F.deconvolution_nd(
            x, W, None, stride=1, pad=1, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        if cuda.cudnn.cudnn.getVersion() >= 4000:
            name = 'cupy.cudnn.cudnn.convolutionBackwardData_v3'
        else:
            name = 'cupy.cudnn.cudnn.convolutionBackwardData_v2'
        with mock.patch(name) as func:
            self.forward()
            self.assertEqual(func.called, self.expected)

    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.convolutionForward') as func:
            y.backward()
            self.assertEqual(func.called, self.expected)


class TestDeconvolutionNDarraySupplied(unittest.TestCase):

    def setUp(self):
        N = 2
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (N, in_channels, 3, 3, 3)
        self.x_data = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        W_shape = (in_channels, out_channels, 1, 1, 1)
        self.W_data = numpy.random.uniform(-1, 1, W_shape).astype(dtype)
        self.b_data = numpy.random.uniform(-1, 1, out_channels).astype(dtype)

    def check_array_supplied(self, x_ary, W_ary, b_ary):
        y_ary = F.deconvolution_nd(x_ary, W_ary, b_ary)

        x_var = chainer.Variable(x_ary)
        W_var = chainer.Variable(W_ary)
        b_var = chainer.Variable(b_ary)
        y_var = F.deconvolution_nd(x_var, W_var, b_var)

        testing.assert_allclose(y_ary.data, y_var.data)

    def test_array_supplied_cpu(self):
        self.check_array_supplied(self.x_data, self.W_data, self.b_data)

    @attr.gpu
    def test_array_supplied_gpu(self):
        self.check_array_supplied(cuda.to_gpu(self.x_data),
                                  cuda.to_gpu(self.W_data),
                                  cuda.to_gpu(self.b_data))


class TestDeconvolutionNDTypeCheck(unittest.TestCase):

    def test_number_of_inputs(self):
        # Too few inputs
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.connection.deconvolution_nd.DeconvolutionND(1)(x)

        # Too much inputs
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.connection.deconvolution_nd.DeconvolutionND(1)(x, W, b, x)

    def test_data_and_weight(self):
        # dtype of data
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.int32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

        # dtype of weight
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.int32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

        # ndim of weight
        x = numpy.random.uniform(-1, 1, (2, 3, 4, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

        # shapes of data and weight
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (2, 2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

    def test_supplied_outsize(self):
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        outsize = (10,)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, outsize=outsize)

    def test_bias(self):
        # dtype
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (2,)).astype(numpy.int32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, b=b)

        # ndim
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, b=b)

        # shape
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, b=b)

    def test_estimated_outsize(self):
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        stride = 1
        pad = 10
        with self.assertRaises(AssertionError):
            F.deconvolution_nd(x, W, stride=stride, pad=pad)


testing.run_module(__name__, __file__)

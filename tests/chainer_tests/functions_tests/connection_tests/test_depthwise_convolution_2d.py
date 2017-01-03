import unittest

import numpy

from chainer import cuda
from chainer.functions.connection import convolution_2d
from chainer.functions.connection import depthwise_convolution_2d
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestDepthwiseConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        channel_multiplier = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 1
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (channel_multiplier, in_channels, kh, kw)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, in_channels * channel_multiplier).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (2, 6, 2, 2)).astype(self.x_dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data, b_data):
        args1 = (x_data, W_data)
        args2 = (x_data, W_data)
        if b_data is not None:
            args1 = args1 + (b_data,)
            b_data = sum(numpy.split(b_data, W_data.shape[1]))
            args2 = args2 + (b_data,)

        f1 = depthwise_convolution_2d.DepthwiseConvolution2D(self.stride,
                                                             self.pad)
        y1 = f1(*args1)
        arys = numpy.split(y1.data, self.W.shape[1], axis=1)
        y1 = sum(arys)

        f2 = convolution_2d.Convolution2DFunction(self.stride, self.pad)
        y2 = f2(*args2).data
        testing.assert_allclose(y1, y2, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b)

    def test_forward_cpu_nobias(self):
        self.check_forward(self.x, self.W, None)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                           cuda.to_gpu(self.b))

    @attr.gpu
    def test_forward_gpu_nobias(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W), None)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            depthwise_convolution_2d.DepthwiseConvolution2D(
                self.stride, self.pad),
            args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)


testing.run_module(__name__, __file__)

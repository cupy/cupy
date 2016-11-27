import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import initializers
from chainer.links import convolution_nd
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
from chainer.utils import conv_nd


@testing.parameterize(*testing.product({
    'dims': [(5,), (5, 4), (4, 3, 3)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestConvolutionND(unittest.TestCase):

    def setUp(self):
        ndim = len(self.dims)
        self.ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim

        self.link = convolution_nd.ConvolutionND(
            ndim, 3, 2, self.ksize, stride=self.stride, pad=self.pad,
            initial_bias=initializers.Uniform(scale=1., dtype=self.dtype))
        self.link.cleargrads()

        x_shape = (2, 3) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        gy_shape = (2, 2) + tuple(
            conv.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(
                self.dims, self.ksize, self.stride, self.pad))
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

        self.check_backward_options = {'eps': 1e-2, 'atol': 1e-3, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4}

    @attr.gpu
    def test_im2col_consistency(self):
        col_cpu = conv_nd.im2col_nd_cpu(
            self.x, self.ksize, self.stride, self.pad)
        col_gpu = conv_nd.im2col_nd_gpu(
            cuda.to_gpu(self.x), self.ksize, self.stride, self.pad)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        col = conv_nd.im2col_nd_cpu(self.x, self.ksize, self.stride, self.pad)
        im_cpu = conv_nd.col2im_nd_cpu(col, self.stride, self.pad, self.dims)
        im_gpu = conv_nd.col2im_nd_gpu(
            cuda.to_gpu(col), self.stride, self.pad, self.dims)
        testing.assert_allclose(im_cpu, im_gpu.get())

    def check_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, self.dtype)

        self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, self.dtype)

        testing.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.cudnn
    @condition.retry(3)
    def test_forward_consistency(self):
        self.check_forward_consistency()

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_im2col(self):
        self.link.use_cudnn = False
        self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b),
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.link.use_cudnn = False
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_pickling(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data1 = y.data

        del x, y

        pickled = pickle.dumps(self.link, -1)
        del self.link
        self.link = pickle.loads(pickled)

        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data2 = y.data

        testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        self.link.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))


class TestConvolutionNDNoInitialBias(unittest.TestCase):

    def test_no_initial_bias(self):
        ndim = 3
        ksize = 3
        link = convolution_nd.ConvolutionND(
            ndim, 3, 2, ksize, initial_bias=None)
        self.assertIsNone(link.b)


testing.run_module(__name__, __file__)

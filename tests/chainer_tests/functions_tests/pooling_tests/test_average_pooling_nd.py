import unittest

import functools
import mock
import numpy
import operator
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
import pooling_nd_helper


@testing.parameterize(*testing.product({
    'dims': [(4,), (4, 3), (4, 3, 2)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestAveragePoolingND(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim

        x_shape = (2, 3) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        outs = tuple(conv.get_conv_outsize(d, k, s, p, False)
                     for (d, k, s, p) in six.moves.zip(
                         self.dims, self.ksize, self.stride, self.pad))
        gy_shape = (2, 3) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'eps': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'eps': 1e-1, 'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, use_cudnn=True):
        dims = self.dims
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        x = chainer.Variable(x_data)
        y = functions.average_pooling_nd(x, ksize, stride, pad,
                                         use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        patches = pooling_nd_helper.pooling_patches(
            dims, ksize, stride, pad, False)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                size = functools.reduce(operator.mul, ksize)
                expect = numpy.array([x[idx].sum() for idx in patches])
                expect = expect.reshape(y_data.shape[2:]) / size
                testing.assert_allclose(
                    expect, y_data[k, c], **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_forward_consistency_regression(self, x_data, use_cudnn=True):
        # Regression test to average_pooling_2d.

        if len(self.dims) != 2:
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad

        y_nd = functions.average_pooling_nd(x_data, ksize, stride=stride,
                                            pad=pad, use_cudnn=use_cudnn)
        y_2d = functions.average_pooling_2d(x_data, ksize, stride=stride,
                                            pad=pad, use_cudnn=use_cudnn)
        testing.assert_allclose(y_nd.data, y_2d.data)

    @condition.retry(3)
    def test_forward_consistency_regression_cpu(self):
        self.check_forward_consistency_regression(self.x)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_consistency_regression_gpu(self):
        self.check_forward_consistency_regression(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_regression_no_cudnn(self):
        self.check_forward_consistency_regression(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.AveragePoolingND(
                self.ndim, self.ksize, self.stride, self.pad,
                use_cudnn=use_cudnn),
            x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)

    def check_backward_consistency_regression(self, x_data, gy_data,
                                              use_cudnn=True):
        # Regression test to two-dimensional average pooling layer.

        if len(self.dims) != 2:
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        xp = cuda.get_array_module(x_data)

        # Backward computation for N-dimensional average pooling layer.
        x_nd = chainer.Variable(xp.array(x_data))
        func_nd = functions.AveragePoolingND(self.ndim, ksize, stride=stride,
                                             pad=pad, use_cudnn=use_cudnn)
        y_nd = func_nd(x_nd)
        y_nd.grad = gy_data
        y_nd.backward()

        # Backward computation for two-dimensional average pooling layer.
        x_2d = chainer.Variable(xp.array(x_data))
        func_2d = functions.AveragePooling2D(ksize, stride=stride, pad=pad,
                                             cover_all=False,
                                             use_cudnn=use_cudnn)
        y_2d = func_2d(x_2d)
        y_2d.grad = gy_data
        y_2d.backward()

        # Test that the two result gradients are close enough.
        testing.assert_allclose(x_nd.grad, x_2d.grad)

    @condition.retry(3)
    def test_backward_consistency_regression_cpu(self):
        self.check_backward_consistency_regression(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_consistency_regression_gpu(self):
        self.check_backward_consistency_regression(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_consistency_regression_no_cudnn(self):
        self.check_backward_consistency_regression(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), use_cudnn=False)


@testing.parameterize(*testing.product({
    'dims': [(4, 3, 2), (3, 2), (2,)],
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestAveragePoolingNDCudnnCall(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.arange(functools.reduce(operator.mul, x_shape),
                                  dtype=self.dtype).reshape(x_shape)
        gy_shape = (2, 3) + tuple(
            conv.get_conv_outsize(d, k, s, p)
            for (d, k, s, p)
            in six.moves.zip(self.dims, self.ksize, self.stride, self.pad))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.average_pooling_nd(
            x, self.ksize, self.stride, self.pad, use_cudnn=self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports average-pooling-nd')
    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.poolingForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn and self.ndim > 1)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports average-pooling-nd')
    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn and self.ndim > 1)


class TestAveragePoolingNDCoverAllNotSupported(unittest.TestCase):

    def test_cover_all_not_supported(self):
        with self.assertRaises(ValueError):
            functions.AveragePoolingND(3, 3, cover_all=True)


testing.run_module(__name__, __file__)

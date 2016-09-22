import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv


class TestDilatedConvolution2D(unittest.TestCase):

    def setUp(self):
        self.link = links.DilatedConvolution2D(
            3, 2, 3, stride=2, pad=2, dilate=2)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()

        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 2, 2, 2)).astype(numpy.float32)

    @attr.gpu
    def test_im2col_consistency(self):
        col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        col_gpu = conv.im2col_gpu(
            cuda.to_gpu(self.x), 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        h, w = self.x.shape[2:]
        im_cpu = conv.col2im_cpu(col, 2, 2, 2, 2, h, w, dy=2, dx=2)
        im_gpu = conv.col2im_gpu(
            cuda.to_gpu(col), 2, 2, 2, 2, h, w, dy=2, dx=2)
        testing.assert_allclose(im_cpu, im_gpu.get())

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
        self.check_forward_consistency()

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_im2col(self):
        self.link.use_cudnn = False
        self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
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


class TestDilatedConvolution2DParameterShapePlaceholder(unittest.TestCase):

    def setUp(self):
        in_channels = None
        self.link = links.DilatedConvolution2D(
            in_channels, 2, 3, stride=2, pad=2, dilate=2)
        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(numpy.float32)
        self.link(chainer.Variable(self.x))
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 2, 2, 2)).astype(numpy.float32)

    @attr.gpu
    def test_im2col_consistency(self):
        col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        col_gpu = conv.im2col_gpu(
            cuda.to_gpu(self.x), 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 2, 2, dy=2, dx=2)
        h, w = self.x.shape[2:]
        im_cpu = conv.col2im_cpu(col, 2, 2, 2, 2, h, w, dy=2, dx=2)
        im_gpu = conv.col2im_gpu(
            cuda.to_gpu(col), 2, 2, 2, 2, h, w, dy=2, dx=2)
        testing.assert_allclose(im_cpu, im_gpu.get())

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
        self.check_forward_consistency()

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_im2col(self):
        self.link.use_cudnn = False
        self.check_forward_consistency()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
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

testing.run_module(__name__, __file__)

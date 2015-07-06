import unittest

import numpy
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr
from chainer.utils import conv


if cuda.available:
    cuda.init()


class TestConvolution2D(unittest.TestCase):

    def setUp(self, use_cudnn=True):
        self.func = functions.Convolution2D(
            3, 2, 3, stride=2, pad=1, use_cudnn=use_cudnn)
        self.func.b = numpy.random.uniform(
            -1, 1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)

        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 2, 2, 2)).astype(numpy.float32)

    @attr.gpu
    def test_im2col_consistency(self):
        col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
        col_gpu = conv.im2col_gpu(cuda.to_gpu(self.x), 3, 3, 2, 2, 1, 1)
        gradient_check.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
        h, w = self.x.shape[2:]
        im_cpu = conv.col2im_cpu(col,         2, 2, 1, 1, h, w)
        im_gpu = conv.col2im_gpu(cuda.to_gpu(col), 2, 2, 1, 1, h, w)
        gradient_check.assert_allclose(im_cpu, im_gpu.get())

    @attr.cudnn
    def test_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.func(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)

        self.func.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.func(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)

        for i in six.moves.range(y_cpu.data.shape[0]):
            print(i, numpy.abs(y_cpu.data[i] - y_gpu.data[i].get()).max())
        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.func.use_cudnn = False
        self.test_forward_consistency()

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, func.W, func.b), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, func.gW)
        gradient_check.assert_allclose(gb, func.gb)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_im2col(self):
        self.func.use_cudnn = False
        self.test_backward_gpu()

    def check_pickling(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y_data1 = y.data

        del x, y

        pickled = pickle.dumps(self.func, -1)
        del self.func
        self.func = pickle.loads(pickled)

        x = chainer.Variable(x_data)
        y = self.func(x)
        y_data2 = y.data

        gradient_check.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        self.func.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))

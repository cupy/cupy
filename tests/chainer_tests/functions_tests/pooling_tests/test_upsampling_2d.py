from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv

import chainer.functions as F
import numpy
import unittest


@testing.parameterize(
    {'in_shape': (4, 3, 6, 8)},
    {'in_shape': (4, 3, 5, 7)},
)
class TestUpsampling2D(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype('f')
        self.p = F.MaxPooling2D(2, 2, use_cudnn=False)
        self.pooled_y = self.p(self.x)
        self.gy = numpy.random.uniform(
            -1, 1, self.in_shape).astype(numpy.float32)

    def check_forward(self, y):
        y = F.upsampling_2d(
            self.pooled_y, self.p.indexes, ksize=(self.p.kh, self.p.kw),
            stride=(self.p.sy, self.p.sx), pad=(self.p.ph, self.p.pw),
            outsize=self.in_shape[2:], cover_all=self.p.cover_all)
        if isinstance(y.data, numpy.ndarray):
            y = conv.im2col_cpu(y.data, self.p.kh, self.p.kw,
                                self.p.sy, self.p.sx, self.p.ph, self.p.pw)
        else:
            y = conv.im2col_gpu(y.data, self.p.kh, self.p.kw,
                                self.p.sy, self.p.sx, self.p.ph, self.p.pw)
        for i in numpy.ndindex(y.shape):
            n, c, ky, kx, oy, ox = i
            up_y = y[n, c, ky, kx, oy, ox]
            if ky * y.shape[3] + kx == self.p.indexes[n, c, oy, ox]:
                in_y = self.pooled_y.data[n, c, oy, ox]
                testing.assert_allclose(in_y, up_y)
            else:
                testing.assert_allclose(up_y, 0)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.pooled_y.to_cpu()
        self.check_forward(self.pooled_y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.pooled_y.to_gpu()
        self.check_forward(self.pooled_y)

    def check_backward(self, x_data, y_grad):
        func = F.Upsampling2D(
            self.p.indexes, ksize=(self.p.kh, self.p.kw),
            stride=(self.p.sy, self.p.sx), pad=(self.p.ph, self.p.pw),
            outsize=self.in_shape[2:], cover_all=self.p.cover_all)
        gradient_check.check_backward(func, x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.pooled_y.data, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(
            self.pooled_y.data), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)

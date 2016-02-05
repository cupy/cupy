import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition


class DetFunctionTestBase(object):

    def setUp(self):
        self.x, self.y, self.gy = self.make_data()
        self.ct = numpy.array(
            [ix.T for ix in self.x.copy()]).astype(numpy.float32)

    def det_transpose(self, gpu=False):
        if gpu:
            cx = cuda.to_gpu(self.x.copy())
            ct = cuda.to_gpu(self.ct.copy())
        else:
            cx = self.x.copy()
            ct = self.ct.copy()
        xn = chainer.Variable(cx)
        xt = chainer.Variable(ct)
        yn = F.batch_det(xn)
        yt = F.batch_det(xt)
        gradient_check.assert_allclose(yn.data, yt.data, rtol=1e-4, atol=1)

    @attr.gpu
    @condition.retry(3)
    def test_det_transpose_gpu(self):
        self.det_transpose(gpu=True)

    @condition.retry(3)
    def test_det_transpose_cpu(self):
        self.det_transpose(gpu=False)

    def det_scaling(self, gpu=False):
        scaling = numpy.random.randn(1).astype('float32')
        if gpu:
            cx = cuda.to_gpu(self.x.copy())
            sx = cuda.to_gpu(scaling * self.x.copy())
        else:
            cx = self.x.copy()
            sx = scaling * self.x.copy()
        c = float(scaling ** self.x.shape[1])
        cxv = chainer.Variable(cx)
        sxv = chainer.Variable(sx)
        cxd = F.batch_det(cxv)
        sxd = F.batch_det(sxv)
        gradient_check.assert_allclose(cxd.data * c, sxd.data)

    @attr.gpu
    @condition.retry(3)
    def test_det_scaling_gpu(self):
        self.det_scaling(gpu=True)

    @condition.retry(3)
    def test_det_scaling_cpu(self):
        self.det_scaling(gpu=False)

    def det_identity(self, gpu=False):
        idt = [numpy.identity(self.x.shape[1]).astype(numpy.float32)
               for _ in range(self.x.shape[0])]
        idt = numpy.array(idt).astype(numpy.float32)
        chk = numpy.ones(self.x.shape[0]).astype(numpy.float32)
        if gpu:
            chk = cuda.to_gpu(chk)
            idt = cuda.to_gpu(idt)
        idtv = chainer.Variable(idt)
        idtd = F.batch_det(idtv)
        gradient_check.assert_allclose(idtd.data, chk, rtol=1e-4, atol=1e-4)

    @attr.gpu
    @condition.retry(3)
    def test_det_identity_gpu(self):
        self.det_identity(gpu=True)

    @condition.retry(3)
    def test_det_identity_cpu(self):
        self.det_identity(gpu=False)

    def det_product(self, gpu=False):
        if gpu:
            cx = cuda.to_gpu(self.x.copy())
            cy = cuda.to_gpu(self.y.copy())
        else:
            cx = self.x.copy()
            cy = self.y.copy()
        vx = chainer.Variable(cx)
        vy = chainer.Variable(cy)
        dxy1 = F.batch_det(F.batch_matmul(vx, vy))
        dxy2 = F.batch_det(vx) * F.batch_det(vy)
        gradient_check.assert_allclose(dxy1.data, dxy2.data, rtol=1e-4,
                                       atol=1e-4)

    @condition.retry(3)
    def test_det_product_cpu(self):
        self.det_product(gpu=False)

    @attr.gpu
    @condition.retry(3)
    def test_det_product_gpu(self):
        self.det_product(gpu=True)

    @attr.gpu
    @condition.retry(3)
    def test_batch_backward_gpu(self):
        x_data = cuda.to_gpu(self.x.copy())
        y_grad = cuda.to_gpu(self.gy.copy())
        gradient_check.check_backward(F.batch_det, x_data, y_grad)

    @condition.retry(3)
    def test_batch_backward_cpu(self):
        x_data, y_grad = self.x.copy(), self.gy.copy()
        gradient_check.check_backward(F.batch_det, x_data, y_grad)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        x_data = cuda.to_gpu(self.x[0].copy())
        y_grad = cuda.to_gpu(self.gy[0][None].copy())
        gradient_check.check_backward(F.det, x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        x_data, y_grad = self.x[0].copy(), self.gy[0][None].copy()
        gradient_check.check_backward(F.det, x_data, y_grad)


class TestSquareMinibatch(DetFunctionTestBase, unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (6, 5, 5)).astype(numpy.float32)
        y = numpy.random.uniform(.5, 1, (6, 5, 5)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (6,)).astype(numpy.float32)
        return x, y, gy

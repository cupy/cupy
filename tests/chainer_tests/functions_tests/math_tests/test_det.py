import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


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
        yn = self.det(xn)
        yt = self.det(xt)
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
        cxd = self.det(cxv)
        sxd = self.det(sxv)
        gradient_check.assert_allclose(cxd.data * c, sxd.data)

    @attr.gpu
    @condition.retry(3)
    def test_det_scaling_gpu(self):
        self.det_scaling(gpu=True)

    @condition.retry(3)
    def test_det_scaling_cpu(self):
        self.det_scaling(gpu=False)

    def det_identity(self, gpu=False):
        if self.x.ndim == 3:
            idt = [numpy.identity(self.x.shape[1]).astype(numpy.float32)
                   for _ in range(self.x.shape[0])]
            idt = numpy.array(idt).astype(numpy.float32)
            chk = numpy.ones(self.x.shape[0]).astype(numpy.float32)
        else:
            idt = numpy.identity(self.x.shape[1]).astype(numpy.float32)
            chk = numpy.ones(1).astype(numpy.float32)
        if gpu:
            chk = cuda.to_gpu(chk)
            idt = cuda.to_gpu(idt)
        idtv = chainer.Variable(idt)
        idtd = self.det(idtv)
        gradient_check.assert_allclose(idtd.data, chk, rtol=1e-4, atol=1e-4)

    @attr.gpu
    def test_det_identity_gpu(self):
        self.det_identity(gpu=True)

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
        dxy1 = self.det(self.matmul(vx, vy))
        dxy2 = self.det(vx) * self.det(vy)
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
        gradient_check.check_backward(self.det, x_data, y_grad)

    @condition.retry(3)
    def test_batch_backward_cpu(self):
        x_data, y_grad = self.x.copy(), self.gy.copy()
        gradient_check.check_backward(self.det, x_data, y_grad)

    def test_expect_scalar_cpu(self):
        x = numpy.random.uniform(.5, 1, (2, 2)).astype(numpy.float32)
        x = chainer.Variable(x)
        y = F.det(x)
        self.assertEqual(y.data.ndim, 1)

    @attr.gpu
    def test_expect_scalar_gpu(self):
        x = cuda.cupy.random.uniform(.5, 1, (2, 2)).astype(numpy.float32)
        x = chainer.Variable(x)
        y = F.det(x)
        self.assertEqual(y.data.ndim, 1)

    def test_zero_det_cpu(self):
        x_data, y_grad = self.x.copy(), self.gy.copy()
        if x_data.ndim == 3:
            x_data[0, :, :] = 0.0
        else:
            x_data[:, :] = 0.0
        with self.assertRaises(numpy.linalg.LinAlgError):
            gradient_check.check_backward(self.det, x_data, y_grad)

    @attr.gpu
    def test_zero_det_gpu(self):
        x_data = cuda.to_gpu(self.x.copy())
        y_grad = cuda.to_gpu(self.gy.copy())
        if x_data.ndim == 3:
            x_data[0, :, :] = 0.0
        else:
            x_data[:, :] = 0.0
        with self.assertRaises(ValueError):
            gradient_check.check_backward(self.det, x_data, y_grad)

    def test_answer_cpu(self):
        for _ in range(5):
            x = numpy.random.uniform(.5, 1, (2, 2)).astype(numpy.float32)
            ans = F.det(chainer.Variable(x)).data
            y = x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
            gradient_check.assert_allclose(ans, y)

    @attr.gpu
    def test_answer_gpu(self):
        for _ in range(5):
            x = cuda.cupy.random.uniform(.5, 1, (2, 2)).astype(numpy.float32)
            ans = F.det(chainer.Variable(x)).data
            y = x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
            gradient_check.assert_allclose(ans, y)

    def check_answer_gpu_cpu(self, shape, repeat=10):
        for _ in range(repeat):
            x = cuda.cupy.random.uniform(.5, 1, shape, dtype='float32')
            gpu = cuda.to_cpu(self.det(chainer.Variable(x)).data)
            cpu = numpy.linalg.det(cuda.to_cpu(x))
            gradient_check.assert_allclose(gpu, cpu)

    @attr.gpu
    def test_answer_gpu_cpu(self):
        if self.batched:
            for w in range(1, 5):
                for s in range(2, 5):
                    self.check_answer_gpu_cpu((w, s, s))
        else:
            w = 1
            for s in range(2, 5):
                self.check_answer_gpu_cpu((s, s))


class TestSquareBatchDet(DetFunctionTestBase, unittest.TestCase):
    batched = True

    def det(self, x):
        return F.batch_det(x)

    def matmul(self, x, y):
        return F.batch_matmul(x, y)

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (6, 3, 3)).astype(numpy.float32)
        y = numpy.random.uniform(.5, 1, (6, 3, 3)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (6,)).astype(numpy.float32)
        return x, y, gy


class TestSquareDet(DetFunctionTestBase, unittest.TestCase):
    batched = False

    def det(self, x):
        return F.det(x)

    def matmul(self, x, y):
        return F.matmul(x, y)

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (5, 5)).astype(numpy.float32)
        y = numpy.random.uniform(.5, 1, (5, 5)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (1,)).astype(numpy.float32)
        return x, y, gy


class DetFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        with self.assertRaises(type_check.InvalidType):
            F.batch_det(chainer.Variable(numpy.zeros((2, 2))))

    def test_invalid_shape(self):
        with self.assertRaises(type_check.InvalidType):
            F.batch_det(chainer.Variable(numpy.zeros((1, 2))))


testing.run_module(__name__, __file__)

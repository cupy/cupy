import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class UnaryFunctionsTestBase(object):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.x, self.gy = self.make_data()

    def check_forward(self, op, op_np, x_data, atol=1e-7, rtol=1e-7):
        x = chainer.Variable(x_data)
        y = op(x)
        gradient_check.assert_allclose(
            op_np(self.x), y.data, atol=atol, rtol=rtol)

    def check_forward_cpu(self, op, op_np, **kwargs):
        self.check_forward(op, op_np, self.x, **kwargs)

    def check_forward_gpu(self, op, op_np, **kwargs):
        self.check_forward(op, op_np, cuda.to_gpu(self.x), **kwargs)

    @condition.retry(3)
    def test_exp_forward_cpu(self):
        self.check_forward_cpu(F.exp, numpy.exp)

    @condition.retry(3)
    def test_log_forward_cpu(self):
        self.check_forward_cpu(F.log, numpy.log)

    @attr.gpu
    @condition.retry(3)
    def test_exp_forward_gpu(self):
        self.check_forward_gpu(F.exp, numpy.exp)

    @attr.gpu
    @condition.retry(3)
    def test_log_forward_gpu(self):
        self.check_forward_gpu(F.log, numpy.log)

    def check_backward(self, op, x_data, y_grad, **kwargs):
        x = chainer.Variable(x_data)
        y = op(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))

        gradient_check.assert_allclose(gx, x.grad, **kwargs)

    def check_backward_cpu(self, op, **kwargs):
        self.check_backward(op, self.x, self.gy, **kwargs)

    def check_backward_gpu(self, op, **kwargs):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                            **kwargs)

    @condition.retry(3)
    def test_exp_backward_cpu(self):
        self.check_backward_cpu(F.exp)

    @condition.retry(3)
    def test_log_backward_cpu(self):
        self.check_backward_cpu(F.log)

    @attr.gpu
    @condition.retry(3)
    def test_exp_backward_gpu(self):
        self.check_backward_gpu(F.exp)

    @attr.gpu
    @condition.retry(3)
    def test_log_backward_gpu(self):
        self.check_backward_gpu(F.log)

    def test_exp(self):
        self.assertEqual(F.Exp().label, 'exp')

    def test_log(self):
        self.assertEqual(F.Log().label, 'log')


class InvFunctionTestBase(UnaryFunctionsTestBase):
    @staticmethod
    def batch_inv(x):
        arr = [numpy.linalg.inv(ix) for ix in x]
        return numpy.array(arr)

    def make_eye_inv(self):
        m = self.x.shape[0]
        n = self.x.shape[1]
        eye = [numpy.eye(n) for _ in range(m)]
        eye = numpy.array(eye).astype('float32')
        return eye

    @condition.retry(3)
    def test_identity_cpu(self):
        x = chainer.Variable(self.x.copy())
        y = F.batch_matmul(x, F.batch_inv(x))
        eye = self.make_eye_inv()
        gradient_check.assert_allclose(y.data, eye, rtol=1e-4, atol=1e-4)

    @attr.gpu
    @condition.retry(3)
    def test_identity_gpu(self):
        eye = self.make_eye_inv()
        x, eye = cuda.to_gpu(self.x.copy()), cuda.to_gpu(eye)
        x = chainer.Variable(x)
        y = F.batch_matmul(x, F.batch_inv(x))
        gradient_check.assert_allclose(y.data, eye, rtol=1e-4, atol=1e-4)

    @condition.retry(3)
    def test_inv_forward_cpu(self):
        self.check_forward_cpu(F.batch_inv, InvFunctionTestBase.batch_inv,
                               atol=1e-5, rtol=1e-5)

    @attr.gpu
    @condition.retry(3)
    def test_inv_forward_gpu(self):
        self.check_forward_gpu(F.batch_inv, InvFunctionTestBase.batch_inv,
                               atol=1e-5, rtol=1e-5)

    @condition.retry(3)
    def test_inv_backward_cpu(self):
        self.check_backward_cpu(F.batch_inv, atol=1e-2, rtol=1e-2)

    @attr.gpu
    @condition.retry(3)
    def test_inv_backward_gpu(self):
        self.check_backward_gpu(F.batch_inv, atol=1e-2, rtol=1e-2)

    def test_inv(self):
        self.assertEqual(F.BatchInv().label, 'inv')


class DetFunctionTestBase(object):

    def setUp(self):
        self.x, self.y, self.gy = self.make_data()
        self.ct = numpy.array([ix.T for ix in self.x.copy()]).astype('f32')

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

    def check_backward(self, x_data, y_grad, op, **kwargs):
        x = chainer.Variable(x_data)
        y = op(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))

        gradient_check.assert_allclose(gx, x.grad, **kwargs)

    @attr.gpu
    @condition.retry(3)
    def test_batch_backward_gpu(self):
        x_data = cuda.to_gpu(self.x.copy())
        y_grad = cuda.to_gpu(self.gy.copy())
        self.check_backward(x_data, y_grad, F.batch_det)

    @condition.retry(3)
    def test_batch_backward_cpu(self):
        x_data, y_grad = self.x.copy(), self.gy.copy()
        self.check_backward(x_data, y_grad, F.batch_det)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        x_data = cuda.to_gpu(self.x[0].copy())
        y_grad = cuda.to_gpu(self.gy[0][None].copy())
        self.check_backward(x_data, y_grad, F.det)

    @condition.retry(3)
    def test_backward_cpu(self):
        x_data, y_grad = self.x[0].copy(), self.gy[0][None].copy()
        self.check_backward(x_data, y_grad, F.det)


class TestUnaryFunctionsSimple(UnaryFunctionsTestBase, unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        return x, gy


class TestUnaryFunctionsZeroDimension(UnaryFunctionsTestBase,
                                      unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        return x, gy


class TestInvMinibatch(InvFunctionTestBase, unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (6, 5, 5)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (6, 5, 5)).astype(numpy.float32)
        return x, gy


class TestSquareMinibatch(DetFunctionTestBase, unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (6, 5, 5)).astype(numpy.float32)
        y = numpy.random.uniform(.5, 1, (6, 5, 5)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (6,)).astype(numpy.float32)
        return x, y, gy

testing.run_module(__name__, __file__)

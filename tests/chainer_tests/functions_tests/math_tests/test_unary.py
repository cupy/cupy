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

testing.run_module(__name__, __file__)

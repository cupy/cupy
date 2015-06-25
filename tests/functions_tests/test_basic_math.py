import unittest

import numpy

from chainer import cuda
from chainer.cuda import to_gpu
import chainer.functions as F
from chainer.gradient_check import assert_allclose
from chainer.gradient_check import numerical_grad
from chainer.testing import attr
from chainer import Variable


if cuda.available:
    cuda.init()


class TestBinaryOp(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_forward(self, op, x1_data, x2_data):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = op(x1, x2)
        if isinstance(y.data, cuda.GPUArray):
            self.assertTrue(hasattr(y.data.gpudata, 'device'))
        assert_allclose(op(self.x1, self.x2), y.data)

    def forward_cpu(self, op):
        self.check_forward(op, self.x1, self.x2)

    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__radd__(x))

    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rsub__(x))

    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rmul__(x))

    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rtruediv__(x))

    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rpow__(x))

    def forward_gpu(self, op):
        self.check_forward(op, to_gpu(self.x1), to_gpu(self.x2))

    @attr.gpu
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__radd__(x))

    @attr.gpu
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rsub__(x))

    @attr.gpu
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rmul__(x))

    @attr.gpu
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rtruediv__(x))

    @attr.gpu
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rpow__(x))

    @attr.gpu
    def test_add_constant_allocation(self):
        x = 0
        y = Variable(cuda.ones((1,)))
        z = y + x
        self.assertEqual(1, z.data.get()[0])
        self.assertTrue(hasattr(z.data.gpudata, 'device'))

    def check_backward(self, op, x1_data, x2_data, y_grad, atol):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = op(x1, x2)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x1.data, x2.data))
        gx1, gx2 = numerical_grad(f, (x1.data, x2.data), (y.grad,))
        assert_allclose(gx1, x1.grad, atol=atol)
        assert_allclose(gx2, x2.grad, atol=atol)

    def backward_cpu(self, op, atol=1e-5):
        self.check_backward(op, self.x1, self.x2, self.gy, atol)

    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y, atol=1e-4)

    def backward_gpu(self, op, atol=1e-5):
        self.check_backward(
            op, to_gpu(self.x1), to_gpu(self.x2), to_gpu(self.gy), atol)

    @attr.gpu
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y, atol=1e-4)


class TestVariableConstantOp(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.value = .5

    def check_forward(self, op, x_data):
        x = Variable(x_data)
        y = op(x, self.value)
        assert_allclose(op(self.x, self.value), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op):
        self.check_forward(op, self.x)

    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y + x)

    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y - x)

    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y * x)

    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y / x)

    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y ** x)

    def forward_gpu(self, op):
        self.check_forward(op, to_gpu(self.x))

    @attr.gpu
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y - x)

    @attr.gpu
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y ** x)

    def check_backward(self, op, x_data, y_grad):
        x = Variable(x_data)
        y = op(x, self.value)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(gx, x.grad)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    def test_radd_backward_cpu(self):
        self.backward_cpu(lambda x, y: y + x)

    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    def test_rsub_backward_cpu(self):
        self.backward_cpu(lambda x, y: y - x)

    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    def test_rmul_backward_cpu(self):
        self.backward_cpu(lambda x, y: y * x)

    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    def test_rdiv_backward_cpu(self):
        self.backward_cpu(lambda x, y: y / x)

    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    def test_rpow_backward_cpu(self):
        self.backward_cpu(lambda x, y: y ** x)

    def backward_gpu(self, op):
        self.check_backward(op, to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_backward_gpu(self):
        self.backward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_rsub_backward_gpu(self):
        self.backward_gpu(lambda x, y: y - x)

    @attr.gpu
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_backward_gpu(self):
        self.backward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_backward_gpu(self):
        self.backward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_backward_gpu(self):
        self.backward_gpu(lambda x, y: y ** x)


class TestVariableConstantArrayOp(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.value = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_forward(self, op, x_data, gpu, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        v = value
        if gpu:
            v = cuda.to_gpu(v)
        x = Variable(x_data)
        y = op(x, v)
        assert_allclose(op(self.x, value), y.data, atol=1e-6, rtol=1e-6)

    def forward_cpu(self, op, positive=False):
        self.check_forward(op, self.x, False, positive)

    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y + x)

    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y - x)

    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y * x)

    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y / x)

    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y ** x, positive=True)

    def forward_gpu(self, op, positive=False):
        self.check_forward(op, to_gpu(self.x), True, positive)

    @attr.gpu
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y - x)

    @attr.gpu
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y ** x, positive=True)

    def check_backward(self, op, x_data, y_grad, gpu, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        if gpu:
            value = cuda.to_gpu(value)
        x = Variable(x_data)
        y = op(x, value)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(gx, x.grad, atol=1e-4, rtol=1e-4)

    def backward_cpu(self, op, positive=False):
        self.check_backward(op, self.x, self.gy, False, positive)

    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    def test_radd_backward_cpu(self):
        self.backward_cpu(lambda x, y: y + x)

    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    def test_rsub_backward_cpu(self):
        self.backward_cpu(lambda x, y: y - x)

    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    def test_rmul_backward_cpu(self):
        self.backward_cpu(lambda x, y: y * x)

    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    def test_rdiv_backward_cpu(self):
        self.backward_cpu(lambda x, y: y / x)

    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    def test_rpow_backward_cpu(self):
        self.backward_cpu(lambda x, y: y ** x, positive=True)

    def backward_gpu(self, op, positive=False):
        self.check_backward(
            op, to_gpu(self.x), to_gpu(self.gy), True, positive)

    @attr.gpu
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_backward_gpu(self):
        self.backward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_backward_gpu(self):
        self.backward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_backward_gpu(self):
        self.backward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_backward_gpu(self):
        self.backward_gpu(lambda x, y: y ** x, positive=True)


class TestUnaryFunctions(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_forward(self, op, op_np, x_data):
        x = Variable(x_data)
        y = op(x)
        assert_allclose(op_np(self.x), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    def test_exp_forward_cpu(self):
        self.forward_cpu(F.exp, numpy.exp)

    def test_log_forward_cpu(self):
        self.forward_cpu(F.log, numpy.log)

    def forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, to_gpu(self.x))

    @attr.gpu
    def test_exp_forward_gpu(self):
        self.forward_gpu(F.exp, numpy.exp)

    @attr.gpu
    def test_log_forward_gpu(self):
        self.forward_gpu(F.log, numpy.log)

    def check_backward(self, op, x_data, y_grad):
        x = Variable(x_data)
        y = op(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = numerical_grad(f, (x.data,), (y.grad,))

        assert_allclose(gx, x.grad)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def test_exp_backward_cpu(self):
        self.backward_cpu(F.exp)

    def test_log_backward_cpu(self):
        self.backward_cpu(F.log)

    def backward_gpu(self, op):
        self.check_backward(op, to_gpu(self.x), to_gpu(self.gy))

    @attr.gpu
    def test_exp_backward_gpu(self):
        self.backward_gpu(F.exp)

    @attr.gpu
    def test_log_backward_gpu(self):
        self.backward_gpu(F.log)

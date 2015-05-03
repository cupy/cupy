from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad

cuda.init()

class TestBinaryOp(TestCase):
    def setUp(self):
        self.x1 = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_forward(self, op, x1_data, x2_data):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = op(x1, x2)
        assert_allclose(op(self.x1, self.x2), y.data, atol=0, rtol=0)

    def forward_cpu(self, op):
        self.check_forward(op, self.x1, self.x2)

    def test_add_forward_cpu(self): self.forward_cpu(lambda x, y: x + y)
    def test_sub_forward_cpu(self): self.forward_cpu(lambda x, y: x - y)
    def test_mul_forward_cpu(self): self.forward_cpu(lambda x, y: x * y)
    def test_div_forward_cpu(self): self.forward_cpu(lambda x, y: x / y)

    def forward_gpu(self, op):
        self.check_forward(op, to_gpu(self.x1), to_gpu(self.x2))

    def test_add_forward_gpu(self): self.forward_gpu(lambda x, y: x + y)
    def test_sub_forward_gpu(self): self.forward_gpu(lambda x, y: x - y)
    def test_mul_forward_gpu(self): self.forward_gpu(lambda x, y: x * y)
    def test_div_forward_gpu(self): self.forward_gpu(lambda x, y: x / y)

    def check_backward(self, op, x1_data, x2_data, y_grad):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = op(x1, x2)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x1.data, x2.data))
        gx1, gx2 = numerical_grad(f, (x1.data, x2.data), (y.grad,))
        assert_allclose(gx1, x1.grad)
        assert_allclose(gx2, x2.grad)

    def backward_cpu(self, op):
        self.check_backward(op, self.x1, self.x2, self.gy)

    def test_add_backward_cpu(self): self.backward_cpu(lambda x, y: x + y)
    def test_sub_backward_cpu(self): self.backward_cpu(lambda x, y: x - y)
    def test_mul_backward_cpu(self): self.backward_cpu(lambda x, y: x * y)
    def test_div_backward_cpu(self): self.backward_cpu(lambda x, y: x / y)

    def backward_gpu(self, op):
        self.check_backward(op, to_gpu(self.x1), to_gpu(self.x2), to_gpu(self.gy))

    def test_add_backward_gpu(self): self.backward_gpu(lambda x, y: x + y)
    def test_sub_backward_gpu(self): self.backward_gpu(lambda x, y: x - y)
    def test_mul_backward_gpu(self): self.backward_gpu(lambda x, y: x * y)
    def test_div_backward_gpu(self): self.backward_gpu(lambda x, y: x / y)


class TestVariableConstantOp(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.value = .5

    def check_forward(self, op, x_data):
        x = Variable(x_data)
        y = op(x, self.value)
        assert_allclose(op(self.x, self.value), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op):
        self.check_forward(op, self.x)

    def test_add_forward_cpu(self):  self.forward_cpu(lambda x, y: x + y)
    def test_radd_forward_cpu(self): self.forward_cpu(lambda x, y: y + x)
    def test_sub_forward_cpu(self):  self.forward_cpu(lambda x, y: x - y)
    def test_rsub_forward_cpu(self): self.forward_cpu(lambda x, y: y - x)
    def test_mul_forward_cpu(self):  self.forward_cpu(lambda x, y: x * y)
    def test_rmul_forward_cpu(self): self.forward_cpu(lambda x, y: y * x)
    def test_div_forward_cpu(self):  self.forward_cpu(lambda x, y: x / y)
    def test_rdiv_forward_cpu(self): self.forward_cpu(lambda x, y: y / x)
    def test_pow_forward_cpu(self):  self.forward_cpu(lambda x, y: x ** y)
    def test_rpow_forward_cpu(self): self.forward_cpu(lambda x, y: y ** x)

    def forward_gpu(self, op):
        self.check_forward(op, to_gpu(self.x))

    def test_add_forward_gpu(self):  self.forward_gpu(lambda x, y: x + y)
    def test_radd_forward_gpu(self): self.forward_gpu(lambda x, y: y + x)
    def test_sub_forward_gpu(self):  self.forward_gpu(lambda x, y: x - y)
    def test_rsub_forward_gpu(self): self.forward_gpu(lambda x, y: y - x)
    def test_mul_forward_gpu(self):  self.forward_gpu(lambda x, y: x * y)
    def test_rmul_forward_gpu(self): self.forward_gpu(lambda x, y: y * x)
    def test_div_forward_gpu(self):  self.forward_gpu(lambda x, y: x / y)
    def test_rdiv_forward_gpu(self): self.forward_gpu(lambda x, y: y / x)
    def test_pow_forward_gpu(self):  self.forward_gpu(lambda x, y: x ** y)
    def test_rpow_forward_gpu(self): self.forward_gpu(lambda x, y: y ** x)

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

    def test_add_backward_cpu(self):  self.backward_cpu(lambda x, y: x + y)
    def test_radd_backward_cpu(self): self.backward_cpu(lambda x, y: y + x)
    def test_sub_backward_cpu(self):  self.backward_cpu(lambda x, y: x - y)
    def test_rsub_backward_cpu(self): self.backward_cpu(lambda x, y: y - x)
    def test_mul_backward_cpu(self):  self.backward_cpu(lambda x, y: x * y)
    def test_rmul_backward_cpu(self): self.backward_cpu(lambda x, y: y * x)
    def test_div_backward_cpu(self):  self.backward_cpu(lambda x, y: x / y)
    def test_rdiv_backward_cpu(self): self.backward_cpu(lambda x, y: y / x)
    def test_pow_backward_cpu(self):  self.backward_cpu(lambda x, y: x ** y)
    def test_rpow_backward_cpu(self): self.backward_cpu(lambda x, y: y ** x)

    def backward_gpu(self, op):
        self.check_backward(op, to_gpu(self.x), to_gpu(self.gy))

    def test_add_backward_gpu(self):  self.backward_gpu(lambda x, y: x + y)
    def test_radd_backward_gpu(self): self.backward_gpu(lambda x, y: y + x)
    def test_sub_backward_gpu(self):  self.backward_gpu(lambda x, y: x - y)
    def test_rsub_backward_gpu(self): self.backward_gpu(lambda x, y: y - x)
    def test_mul_backward_gpu(self):  self.backward_gpu(lambda x, y: x * y)
    def test_rmul_backward_gpu(self): self.backward_gpu(lambda x, y: y * x)
    def test_div_backward_gpu(self):  self.backward_gpu(lambda x, y: x / y)
    def test_rdiv_backward_gpu(self): self.backward_gpu(lambda x, y: y / x)
    def test_pow_backward_gpu(self):  self.backward_gpu(lambda x, y: x ** y)
    def test_rpow_backward_gpu(self): self.backward_gpu(lambda x, y: y ** x)

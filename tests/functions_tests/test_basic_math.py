from unittest import TestCase

import numpy
import pycuda.gpuarray as gpuarray

from chain import Variable
from chain.gradient_check import numerical_grad, l_infty_dist

class TestBinaryOp(TestCase):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    def forward_cpu(self, op):
        x1 = Variable(self.x1)
        x2 = Variable(self.x2)
        y = op(x1, x2)
        self.assertTrue((op(self.x1, self.x2) == y.data).all())

    def test_add_forward_cpu(self): self.forward_cpu(lambda x, y: x + y)
    def test_sub_forward_cpu(self): self.forward_cpu(lambda x, y: x - y)
    def test_mul_forward_cpu(self): self.forward_cpu(lambda x, y: x * y)
    def test_div_forward_cpu(self): self.forward_cpu(lambda x, y: x / y)

    def forward_gpu(self, op):
        x1 = Variable(gpuarray.to_gpu(self.x1))
        x2 = Variable(gpuarray.to_gpu(self.x2))
        y = op(x1, x2)
        self.assertTrue((op(self.x1, self.x2) == y.data.get()).all())

    def test_add_forward_gpu(self): self.forward_gpu(lambda x, y: x + y)
    def test_sub_forward_gpu(self): self.forward_gpu(lambda x, y: x - y)
    def test_mul_forward_gpu(self): self.forward_gpu(lambda x, y: x * y)
    def test_div_forward_gpu(self): self.forward_gpu(lambda x, y: x / y)

    def backward_cpu(self, op):
        x1 = Variable(self.x1)
        x2 = Variable(self.x2)
        y = op(x1, x2)
        y.grad = self.gy
        y.backward()

        func = y.creator
        f = lambda: func.forward((x1.data, x2.data))
        gx1, gx2 = numerical_grad(f, (x1.data, x2.data), (y.grad,))

        self.assertLess(l_infty_dist(gx1, x1.grad), 1e-5)
        self.assertLess(l_infty_dist(gx2, x2.grad), 1e-5)

    def test_add_backward_cpu(self): self.backward_cpu(lambda x, y: x + y)
    def test_sub_backward_cpu(self): self.backward_cpu(lambda x, y: x - y)
    def test_mul_backward_cpu(self): self.backward_cpu(lambda x, y: x * y)
    def test_div_backward_cpu(self): self.backward_cpu(lambda x, y: x / y)

    def backward_gpu(self, op):
        x1 = Variable(gpuarray.to_gpu(self.x1))
        x2 = Variable(gpuarray.to_gpu(self.x2))
        y = op(x1, x2)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()

        func = y.creator
        f = lambda: func.forward((x1.data, x2.data))
        gx1, gx2 = numerical_grad(f, (x1.data, x2.data), (y.grad,))

        self.assertLess(l_infty_dist(gx1, x1.grad), 1e-5)
        self.assertLess(l_infty_dist(gx2, x2.grad), 1e-5)

    def test_add_backward_gpu(self): self.backward_gpu(lambda x, y: x + y)
    def test_sub_backward_gpu(self): self.backward_gpu(lambda x, y: x - y)
    def test_mul_backward_gpu(self): self.backward_gpu(lambda x, y: x * y)
    def test_div_backward_gpu(self): self.backward_gpu(lambda x, y: x / y)


class TestVariableConstantOp(TestCase):
    def setUp(self):
        self.x  = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)
        self.value = .5

    def forward_cpu(self, op):
        x = Variable(self.x)
        y = op(x, self.value)
        self.assertTrue((op(self.x, self.value) == y.data).all())

    def test_add_forward_cpu(self):  self.forward_cpu(lambda x, y: x + y)
    def test_radd_forward_cpu(self): self.forward_cpu(lambda x, y: y + x)
    def test_sub_forward_cpu(self):  self.forward_cpu(lambda x, y: x - y)
    def test_rsub_forward_cpu(self): self.forward_cpu(lambda x, y: y - x)
    def test_mul_forward_cpu(self):  self.forward_cpu(lambda x, y: x * y)
    def test_rmul_forward_cpu(self): self.forward_cpu(lambda x, y: y * x)
    def test_div_forward_cpu(self):  self.forward_cpu(lambda x, y: x / y)
    def test_rdiv_forward_cpu(self): self.forward_cpu(lambda x, y: y / x)

    def forward_gpu(self, op):
        x = Variable(gpuarray.to_gpu(self.x))
        y = op(x, self.value)
        self.assertTrue((op(self.x, self.value) == y.data.get()).all())

    def test_add_forward_gpu(self):  self.forward_gpu(lambda x, y: x + y)
    def test_radd_forward_gpu(self): self.forward_gpu(lambda x, y: y + x)
    def test_sub_forward_gpu(self):  self.forward_gpu(lambda x, y: x - y)
    def test_rsub_forward_gpu(self): self.forward_gpu(lambda x, y: y - x)
    def test_mul_forward_gpu(self):  self.forward_gpu(lambda x, y: x * y)
    def test_rmul_forward_gpu(self): self.forward_gpu(lambda x, y: y * x)
    def test_div_forward_gpu(self):  self.forward_gpu(lambda x, y: x / y)
    def test_rdiv_forward_gpu(self): self.forward_gpu(lambda x, y: y / x)

    def backward_cpu(self, op):
        x = Variable(self.x)
        y = op(x, self.value)
        y.grad = self.gy
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

    def test_add_backward_cpu(self):  self.backward_cpu(lambda x, y: x + y)
    def test_radd_backward_cpu(self): self.backward_cpu(lambda x, y: y + x)
    def test_sub_backward_cpu(self):  self.backward_cpu(lambda x, y: x - y)
    def test_rsub_backward_cpu(self): self.backward_cpu(lambda x, y: y - x)
    def test_mul_backward_cpu(self):  self.backward_cpu(lambda x, y: x * y)
    def test_rmul_backward_cpu(self): self.backward_cpu(lambda x, y: y * x)
    def test_div_backward_cpu(self):  self.backward_cpu(lambda x, y: x / y)
    def test_rdiv_backward_cpu(self): self.backward_cpu(lambda x, y: y / x)

    def backward_gpu(self, op):
        x = Variable(gpuarray.to_gpu(self.x))
        y = op(x, self.value)
        y.grad = gpuarray.to_gpu(self.gy)
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx = numerical_grad(f, (x.data,), (y.grad,))

        self.assertLess(l_infty_dist(gx, x.grad), 1e-5)

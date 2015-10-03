import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestSwapaxes(unittest.TestCase):
    axis1 = 0
    axis2 = 1

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.gy = numpy.random.uniform(-1, 1, (3, 4, 2))

    def check_forward(self, x_data):
        axis1, axis2 = self.axis1, self.axis2
        x = chainer.Variable(x_data)
        y = functions.swapaxes(x, axis1, axis2)
        self.assertEqual(y.data.dtype, numpy.float)
        self.assertTrue((self.x.swapaxes(axis1, axis2) ==
                         cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = functions.swapaxes(x, self.axis1, self.axis2)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data.copy(),))

        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,), eps=1e-5)
        gradient_check.assert_allclose(gx, x.grad, rtol=1e-5)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

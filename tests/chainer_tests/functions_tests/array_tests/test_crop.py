import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestCrop(unittest.TestCase):
    axes = [1, 2]

    def setUp(self):
        self.x0_data = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.x1_data = numpy.random.uniform(-1, 1, (4, 2, 1))
        self.gy_data = numpy.random.uniform(-1, 1, (4, 2, 1))

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        y = functions.crop(x0, x1, self.axes)
        self.assertEqual(y.data.dtype, numpy.float)
        numpy.testing.assert_equal(cuda.to_cpu(x0_data)[:, :2, :1],
                                   cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x0_data, self.x1_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0_data),
                           cuda.to_gpu(self.x1_data))

    def check_backward(self, x0_data, x1_data, y_grad):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        y = functions.crop(x0, x1, self.axes)
        y.grad = y_grad
        y.backward()

        xs = (x0.data, x1.data)

        def f():
            func = y.creator
            return func.forward(xs)

        gx, _ = gradient_check.numerical_grad(f, xs, (y.grad,))
        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x0.grad))

    def test_backward_cpu(self):
        self.check_backward(self.x0_data, self.x1_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0_data),
                            cuda.to_gpu(self.x1_data),
                            cuda.to_gpu(self.gy_data))


testing.run_module(__name__, __file__)

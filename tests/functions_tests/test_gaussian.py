import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.m = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.v = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_backward(self, m_data, v_data, y_grad):
        m = chainer.Variable(m_data)
        v = chainer.Variable(v_data)
        y = functions.gaussian(m, v)
        self.assertEqual(y.data.dtype, numpy.float32)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((m.data, v.data))
        gm, gv = gradient_check.numerical_grad(f, (m.data, v.data), (y.grad,))

        gradient_check.assert_allclose(gm, m.grad)
        gradient_check.assert_allclose(gv, v.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.m, self.v, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.m),
                            cuda.to_gpu(self.v),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

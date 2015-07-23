import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


class ConcatTestBase(object):

    def check_forward(self, xs_data, y_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        gradient_check.assert_allclose(y_data, y.data, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(self.xs, self.y, axis=self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            [cuda.to_gpu(x.copy()) for x in self.xs],
            cuda.to_gpu(self.y), axis=self.axis)

    def check_backward(self, xs_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        y.grad = y.data
        y.backward()

        for x in xs:
            gradient_check.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.xs, axis=self.axis)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward([cuda.to_gpu(x.copy()) for x in self.xs],
                            axis=self.axis)


class TestConcat1(unittest.TestCase, ConcatTestBase):

    def setUp(self):
        self.y = numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)
        self.xs = [self.y[:, :2], self.y[:, 2:5], self.y[:, 5:]]
        self.axis = 1


class TestConcat2(unittest.TestCase, ConcatTestBase):

    def setUp(self):
        self.y = numpy.arange(21, dtype=numpy.float32).reshape(7, 3)
        self.xs = [self.y[:2], self.y[2:5], self.y[5:]]
        self.axis = 0


class TestConcatLastAxis(unittest.TestCase, ConcatTestBase):

    def setUp(self):
        self.y = numpy.arange(2, dtype=numpy.float32)
        self.xs = [self.y[:1], self.y[1:]]
        self.axis = 0


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import functions
from chainer.testing import attr
from chainer.utils import type_check


class TestBroadcast(unittest.TestCase):

    x_shape = (3, 1, 5)
    y_shape = (1, 2, 5)
    out_shape = (3, 2, 5)

    def setUp(self):
        uniform = numpy.random.uniform
        self.x_data = uniform(0, 1, self.x_shape).astype(numpy.float32)
        self.y_data = uniform(0, 1, self.y_shape).astype(numpy.float32)
        self.gbx_data = uniform(0, 1, self.out_shape).astype(numpy.float32)
        self.gby_data = uniform(0, 1, self.out_shape).astype(numpy.float32)

    def check_forward(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        bx, by = functions.broadcast(x, y)

        self.assertEqual(bx.data.shape, self.out_shape)
        self.assertEqual(by.data.shape, self.out_shape)

    def test_forward_cpu(self):
        self.check_forward(self.x_data, self.y_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.y_data))

    def check_backward(self, x_data, y_data, gbx_data, gby_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        bx, by = functions.broadcast(x, y)

        func = by.creator
        f = lambda: func.forward((x.data, y.data))

        bx.grad = gbx_data
        bx.backward()
        gx, gy = gradient_check.numerical_grad(
            f, (x.data, y.data), (bx.grad, by.grad))
        gradient_check.assert_allclose(gx, x.grad)

        by.grad = gby_data
        by.backward()
        gx, gy = gradient_check.numerical_grad(
            f, (x.data, y.data), (bx.grad, by.grad))
        gradient_check.assert_allclose(gy, y.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.y_data,
                            self.gbx_data, self.gby_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.y_data),
                            cuda.to_gpu(self.gbx_data),
                            cuda.to_gpu(self.gby_data))


class TestBroadcastFill(TestBroadcast):

    x_shape = (3, 2, 5)
    y_shape = (5,)
    out_shape = (3, 2, 5)


class TestBroadcastScalar(TestBroadcast):

    x_shape = (3, 2, 5)
    y_shape = ()
    out_shape = (3, 2, 5)


class TestBroadcastTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.int32)
        y_data = numpy.zeros((1, 3, 4), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            functions.broadcast(x, y)

    def test_invalid_shape_fill(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.int32)
        y_data = numpy.zeros((4), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            functions.broadcast(x, y)

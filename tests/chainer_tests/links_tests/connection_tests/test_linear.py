import os
import tempfile
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class TestLinear(unittest.TestCase):

    in_shape = (3,)
    out_size = 2

    def setUp(self):
        in_size = numpy.prod(self.in_shape)
        self.link = links.Linear(in_size, self.out_size)
        W = self.link.W.data
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.zerograds()

        self.W = W.copy()  # fixed on CPU
        self.b = b.copy()  # fixed on CPU

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y = self.x.reshape(4, -1).dot(W.T) + b

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        testing.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestLinearParameterShapePlaceholder(unittest.TestCase):

    in_size = 3
    in_shape = (in_size,)
    out_size = 2
    in_size_or_none = None

    def setUp(self):
        self.link = links.Linear(self.in_size_or_none, self.out_size)
        temp_x = numpy.random.uniform(-1, 1,
                                      (self.out_size,
                                       self.in_size)).astype(numpy.float32)
        self.link(chainer.Variable(temp_x))
        W = self.link.W.data
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.zerograds()

        self.W = W.copy()  # fixed on CPU
        self.b = b.copy()  # fixed on CPU

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y = self.x.reshape(4, -1).dot(W.T) + b

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_serialization(self):
        lin1 = links.Linear(None, self.out_size)
        x = chainer.Variable(self.x)
        # Must call the link to initialize weights.
        lin1(x)
        w1 = lin1.W.data
        fd, temp_file_path = tempfile.mkstemp()
        os.close(fd)
        npz.save_npz(temp_file_path, lin1)
        lin2 = links.Linear(None, self.out_size)
        npz.load_npz(temp_file_path, lin2)
        w2 = lin2.W.data
        self.assertEqual((w1 == w2).all(), True)


class TestLinearWithSpatialDimensions(TestLinear):

    in_shape = (3, 2, 2)


class TestInvalidLinear(unittest.TestCase):

    def setUp(self):
        self.link = links.Linear(3, 2)
        self.x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(self.x))


testing.run_module(__name__, __file__)

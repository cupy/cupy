import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'shape': [(9, 11), (99,)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBinaryAccuracy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.t = numpy.random.randint(-1, 2, self.shape).astype(numpy.int32)
        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.binary_accuracy(x, t)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual((), y.data.shape)

        count = 0
        correct = 0
        x_flatten = self.x.ravel()
        t_flatten = self.t.ravel()
        for i in six.moves.range(t_flatten.size):
            if t_flatten[i] == -1:
                continue
            pred = int(x_flatten[i] >= 0)
            if pred == t_flatten[i]:
                correct += 1
            count += 1
        expected = float(correct) / count
        testing.assert_allclose(
            expected, cuda.to_cpu(y.data), **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBinaryAccuracyIgnoreAll(unittest.TestCase):

    def setUp(self):
        shape = (5, 4)
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.t = -numpy.ones(shape).astype(numpy.int32)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.binary_accuracy(x, t)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = 0.0
        testing.assert_allclose(expected, cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


class TestBinaryAccuracyTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x = chainer.Variable(numpy.zeros((3, 2, 5), dtype=numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 3, 5), dtype=numpy.int32))

        with self.assertRaises(type_check.InvalidType):
            chainer.functions.binary_accuracy(x, t)

    def test_invalid_type(self):
        x = chainer.Variable(numpy.zeros((3, 2, 5), dtype=numpy.float32))
        t = chainer.Variable(numpy.zeros((3, 2, 5), dtype=numpy.float32))

        with self.assertRaises(type_check.InvalidType):
            chainer.functions.binary_accuracy(x, t)


testing.run_module(__name__, __file__)

import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(
    {'x_shape': (10, 3), 't_shape': (10,)},
    {'x_shape': (10, 3, 1), 't_shape': (10,)},
    {'x_shape': (10, 3, 1, 1), 't_shape': (10,)},
    {'x_shape': (10, 3, 5), 't_shape': (10, 5)},
    {'x_shape': (10, 3, 5, 4), 't_shape': (10, 5, 4)},
    {'x_shape': (10, 3, 5, 4, 1), 't_shape': (10, 5, 4)},
    {'x_shape': (10, 3, 5, 4, 1, 1), 't_shape': (10, 5, 4)},
)
class TestAccuracy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1,
                                      self.x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=self.t_shape).astype(numpy.int32)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.accuracy(x, t)
        self.assertEqual(y.data.dtype, numpy.float32)
        self.assertEqual((), y.data.shape)

        x_ = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(self.t.size, -1)
        t_ = self.t.ravel()
        count = 0
        for i in six.moves.range(t_.size):
            pred = x_[i].argmax()
            if pred == t_[i]:
                count += 1

        expected = float(count) / self.t.size
        gradient_check.assert_allclose(expected, cuda.to_cpu(y.data))

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


@testing.parameterize(
    {'x_shape': (10, 3), 't_shape': (4,)},
    {'x_shape': (10, 3, 2), 't_shape': (10,)},
    {'x_shape': (10, 3, 1, 2), 't_shape': (10,)},
    {'x_shape': (10, 3, 4), 't_shape': (10, 5)},
    {'x_shape': (10, 3, 5, 2), 't_shape': (10, 5)},
    {'x_shape': (10, 3, 5, 1, 2), 't_shape': (10, 5)},
)
class TestInvalidShape(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1,
                                      self.x_shape).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=self.t_shape).astype(numpy.int32)

    def check_invalid_shape(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        with self.assertRaises(type_check.InvalidType):
            chainer.functions.accuracy(x, t)

    def test_invalid_shape_cpu(self):
        self.check_invalid_shape(numpy)

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.check_invalid_shape(cuda.cupy)


testing.run_module(__name__, __file__)

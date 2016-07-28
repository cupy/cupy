import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


def accuracy(x, t, ignore_label):
    x_ = numpy.rollaxis(x, 1, x.ndim).reshape(t.size, -1)
    t_ = t.ravel()

    if ignore_label is not None:
        count = 0
        for i in six.moves.range(t_.size):
            pred = x_[i].argmax()
            if t_[i] != ignore_label and pred == t_[i]:
                count += 1
        total = (t_ != ignore_label).sum()
    else:
        count = 0
        for i in six.moves.range(t_.size):
            pred = x_[i].argmax()
            if pred == t_[i]:
                count += 1
        total = t_.size

    if total == 0:
        return 0.0
    else:
        return float(count) / total


@testing.parameterize(
    *testing.product_dict(
        [{'x_shape': (10, 3), 't_shape': (10,)},
         {'x_shape': (10, 3, 1), 't_shape': (10,)},
         {'x_shape': (10, 3, 1, 1), 't_shape': (10,)},
         {'x_shape': (10, 3, 5), 't_shape': (10, 5)},
         {'x_shape': (10, 3, 5, 4), 't_shape': (10, 5, 4)},
         {'x_shape': (10, 3, 5, 4, 1), 't_shape': (10, 5, 4)},
         {'x_shape': (10, 3, 5, 4, 1, 1), 't_shape': (10, 5, 4)}],
        [{'ignore_label': None, 't_data': 'randint'},
         {'ignore_label': 0, 't_data': 'randint'},
         {'ignore_label': 0, 't_data': 'zero'}],
        [{'dtype': numpy.float16},
         {'dtype': numpy.float32},
         {'dtype': numpy.float64}]
    )
)
class TestAccuracy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        if self.t_data == 'randint':
            self.t = numpy.random.randint(
                3, size=self.t_shape).astype(numpy.int32)
        elif self.t_data == 'zero':
            self.t = numpy.zeros(self.t_shape).astype(numpy.int32)
        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.accuracy(x, t, self.ignore_label)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual((), y.data.shape)

        expected = accuracy(self.x, self.t, self.ignore_label)
        testing.assert_allclose(
            expected, cuda.to_cpu(y.data), **self.check_forward_options)

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

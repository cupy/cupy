import unittest

import numpy

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


def r2_score(pred, true, sample_weight=None, multioutput="uniform_average"):
    SS_res = numpy.sum((pred - true) ** 2, axis=0)
    SS_tot = numpy.sum((true - numpy.mean(true, axis=0)) ** 2, axis=0)

    if multioutput == 'uniform_average':
        if numpy.any(SS_tot == 0):
            return 0.0
        else:
            return (1 - SS_res / SS_tot).mean()
    elif multioutput == 'raw_values':
        if numpy.any(SS_tot == 0):
            return numpy.where(SS_tot != 0, 1 - SS_res / SS_tot, 0.0)
        else:
            return 1 - SS_res / SS_tot


@testing.parameterize(
    *testing.product_dict(
        [{'x_shape': (10,), 't_shape': (10,)},
         {'x_shape': (10, 1), 't_shape': (10, 1)},
         {'x_shape': (10, 5), 't_shape': (10, 5)},
         {'x_shape': (10, 5, 4), 't_shape': (10, 5, 4)}],
        [{'t_input': 'random'}, {'t_input': 'zero'}],
        [{'multioutput': "uniform_average"},
         {'multioutput': "raw_values"}],
        [{'sample_weight': None}],
        [{'dtype': numpy.float16},
         {'dtype': numpy.float32},
         {'dtype': numpy.float64}]
    )
)
class TestAccuracy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)

        if self.t_input == 'random':
            self.t = numpy.random.uniform(-1, 1, self.t_shape)\
                .astype(self.dtype)
        else:
            self.t = numpy.zeros(self.t_shape).astype(self.dtype)

        self.check_forward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = chainer.functions.r2_score(x, t, self.sample_weight,
                                       self.multioutput)
        self.assertEqual(y.data.dtype, self.dtype)
        if self.multioutput == 'uniform_average':
            self.assertEqual((), y.data.shape)
        elif self.multioutput == 'raw_values':
            self.assertEqual(x_data.shape[1:], y.data.shape)

        expected = r2_score(self.x, self.t, sample_weight=None,
                            multioutput=self.multioutput)
        testing.assert_allclose(
            expected, y.data, **self.check_forward_options)

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

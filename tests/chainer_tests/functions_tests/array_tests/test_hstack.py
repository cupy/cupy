import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (3, 4), 'y_shape': (3, 8)},
        {'shape': (3), 'y_shape': (6,)},
        {'shape': (), 'y_shape': (2,)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestHstack(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
        ]
        self.g = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)

    def check_forward(self, xs_data):
        xs = [chainer.Variable(x) for x in xs_data]
        y = functions.hstack(xs)

        expect = numpy.hstack(self.xs)
        testing.assert_allclose(y.data, expect)

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, xs_data, g_data):
        def func(*xs):
            return functions.hstack(xs)

        gradient_check.check_backward(
            func, xs_data, g_data, eps=2.0 ** -2, atol=1e-3, rtol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(x) for x in self.xs], cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

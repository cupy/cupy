import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'axis': None, 'out_shape': (3,)},
        {'axis': 1, 'out_shape': (1, 3, 1)},
        {'axis': -3, 'out_shape': (1, 3, 1)},
        {'axis': (0, 1, 3), 'out_shape': (3,)},
        {'axis': (3, 1, 0), 'out_shape': (3,)},
        {'axis': (-4, -3, -1), 'out_shape': (3,)},
        {'axis': (-1, -3, -4), 'out_shape': (3,)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestSqueeze(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 1, 3, 1)).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.out_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 2 ** -4, 'rtol': 2 ** -4}

    def check_forward(self, x_data):
        y = functions.squeeze(x_data, axis=self.axis)
        expected = numpy.squeeze(self.x, axis=self.axis)
        testing.assert_allclose(y.data, expected, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        gradient_check.check_backward(
            functions.Squeeze(self.axis),
            x_data, g_data, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))


@testing.parameterize(*testing.product(
    {'axis': [1, (1,)]},
))
class TestSqueezeValueError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 3, 1)).astype('f')

    def check_invalid_type(self, x_data):
        with self.assertRaises(ValueError):
            functions.squeeze(x_data, axis=self.axis)

    def test_invalid_type_cpu(self):
        self.check_invalid_type(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_invalid_type(cuda.to_gpu(self.x))


@testing.parameterize(*testing.product(
    {'axis': [3, -4, (3,), (-4,)]},
))
class TestSqueezeInvalidType(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 3, 1)).astype('f')

    def check_invalid_type(self, x_data):
        with self.assertRaises(type_check.InvalidType):
            functions.squeeze(x_data, axis=self.axis)

    def test_invalid_type_cpu(self):
        self.check_invalid_type(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_invalid_type(cuda.to_gpu(self.x))


class TestSqueezeTypeError(unittest.TestCase):

    def test_invalid_axis(self):
        with self.assertRaises(TypeError):
            functions.Squeeze('a')


testing.run_module(__name__, __file__)

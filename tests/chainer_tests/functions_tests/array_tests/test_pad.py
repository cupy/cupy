import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (), 'pad_width': 1, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': 1, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': (1, 2), 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': ((1, 2), (3, 4)), 'mode': 'constant'},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestPadDefault(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_shape = numpy.pad(self.x, self.pad_width, self.mode).shape
        self.g = numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({
                'atol': 2 ** -6, 'rtol': 2 ** -6})

    def check_forward(self, x_data):
        y = functions.pad(x_data, self.pad_width, self.mode)
        y_expected = numpy.pad(self.x, self.pad_width, self.mode)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(y.data, y_expected)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        gradient_check.check_backward(
            functions.Pad(self.pad_width, self.mode), x_data, g_data,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3), 'pad_width': 1, 'mode': 'constant',
         'constant_values': 1},
        {'shape': (2, 3), 'pad_width': (1, 2), 'mode': 'constant',
         'constant_values': (1, 2)},
        {'shape': (2, 3), 'pad_width': ((1, 2), (3, 4)), 'mode': 'constant',
         'constant_values': ((1, 2), (3, 4))},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
# Old numpy does not work with multi-dimensional constant_values
@testing.with_requires('numpy>=1.11.1')
class TestPad(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_shape = numpy.pad(self.x, self.pad_width, mode=self.mode,
                              constant_values=self.constant_values).shape
        self.g = numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'atol': 2 ** -6, 'rtol': 2 ** -6}

    def check_forward(self, x_data):
        y = functions.pad(x_data, self.pad_width, mode=self.mode,
                          constant_values=self.constant_values)
        y_expected = numpy.pad(self.x, self.pad_width, mode=self.mode,
                               constant_values=self.constant_values)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(y.data, y_expected)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        gradient_check.check_backward(
            functions.Pad(self.pad_width, mode=self.mode,
                          constant_values=self.constant_values),
            x_data, g_data, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))


class TestPadInvalidType(unittest.TestCase):

    def test_invalid_type(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            functions.Pad(1, mode='constant')(x, x)


testing.run_module(__name__, __file__)

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

        # self.check_forward_options = {}
        # self.check_backward_options = {'dtype': numpy.float64}
        # if self.dtype == numpy.float16:
        #     self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
        #     self.check_backward_options = {
        #         'dtype': numpy.float64, 'atol': 2 ** -4, 'rtol': 2 ** -4}

    def check_forward(self, x_data):
        y = functions.pad(x_data, self.pad_width, self.mode)
        y_expected = numpy.pad(self.x, self.pad_width, self.mode)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(y.data, y_expected)
        # testing.assert_allclose(
        #     y.data, y_expected, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        gradient_check.check_backward(
            functions.Pad(self.pad_width, self.mode), x_data, g_data)
        # gradient_check.check_backward(
        #     functions.Tile(self.reps), x_data, g_data,
        #     **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))

# @testing.parameterize(*testing.product({
#     'reps': [-1, (-1, -1)],
# }))
# class TestTileValueError(unittest.TestCase):
#
#     def test_value_error(self):
#         x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
#         with self.assertRaises(ValueError):
#             functions.tile(x, self.reps)
# 
# 
# class TestTileInvalidType(unittest.TestCase):
# 
#     def test_invalid_type(self):
#         x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
#         with self.assertRaises(type_check.InvalidType):
#             functions.Tile(2)(x, x)
# 
# 
# class TestTileTypeError(unittest.TestCase):
# 
#     def test_type_error(self):
#         x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
#         with self.assertRaises(TypeError):
#             functions.tile(x, 'a')


testing.run_module(__name__, __file__)

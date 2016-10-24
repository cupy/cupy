import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import parameterize
from chainer.utils import type_check


@parameterize(
    {'axes': [1, 2], 'offsets': 0, 'sliced_shape': (4, 2, 1)},
    {'axes': [1, 2], 'offsets': [0, 1, 1], 'sliced_shape': (4, 2, 1)},
    {'axes': 1, 'offsets': 1, 'sliced_shape': (4, 2, 2)},
    {'axes': 1, 'offsets': [0, 1, 1], 'sliced_shape': (4, 2, 2)},
    {'axes': [], 'offsets': 0, 'new_axes': 0, 'sliced_shape': (1, 4, 3, 2)},
    {'axes': [], 'offsets': 0, 'new_axes': 2, 'sliced_shape': (4, 3, 1, 2)},
    {'axes': [], 'offsets': 0, 'new_axes': 3, 'sliced_shape': (4, 3, 2, 1)},
    {'slices': (1, -1, 0), 'sliced_shape': ()},
    {'slices': (1, -1), 'sliced_shape': (2,)},
    {'slices': (1, Ellipsis, -1), 'sliced_shape': (3,)},
    {'slices': (1, None, Ellipsis, None, -1), 'sliced_shape': (1, 3, 1)},
)
class TestGetItem(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.shape = (4, 2, 1)
        self.gy_data = numpy.random.uniform(-1, 1, self.sliced_shape)

        if not hasattr(self, 'slices'):
            # Convert axes, offsets and shape to slices
            if isinstance(self.offsets, int):
                self.offsets = tuple([self.offsets] * len(self.shape))
            if isinstance(self.axes, int):
                self.axes = tuple([self.axes])

            self.slices = [slice(None)] * len(self.shape)
            for axis in self.axes:
                self.slices[axis] = slice(
                    self.offsets[axis], self.offsets[axis] + self.shape[axis])

            if hasattr(self, 'new_axes'):
                self.slices.insert(self.new_axes, None)

            self.slices = tuple(self.slices)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.get_item(x, self.slices)
        self.assertEqual(y.data.dtype, numpy.float)
        numpy.testing.assert_equal(cuda.to_cpu(x_data)[self.slices],
                                   cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(functions.GetItem(self.slices),
                                      (x_data,), y_grad)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.gy_data))


class TestInvalidGetItem(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2))

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_advanced_indexing(self):
        with self.assertRaises(ValueError):
            functions.get_item(self.x_data, ([0, 0, 0],))

    def test_multiple_ellipsis(self):
        with self.assertRaises(ValueError):
            functions.get_item(self.x_data, (Ellipsis, Ellipsis))

    def test_too_many_indices(self):
        with self.assertRaises(type_check.InvalidType):
            functions.get_item(self.x_data, (0, 0, 0, 0))


testing.run_module(__name__, __file__)

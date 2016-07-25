import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (7, 3), 'axis': 0, 'ys_section': [2, 5],
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2, 9, 3), 'axis': 1, 'ys_section': 3,
         'slices': [[slice(None), slice(None, 3)], [slice(None), slice(3, 6)],
                    [slice(None), slice(6, None)]]},
        {'shape': (2, 6, 3), 'axis': 1, 'ys_section': 3,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 4)],
                    [slice(None), slice(4, None)]]},
        {'shape': (2,), 'axis': 0, 'ys_section': [1],
         'slices': [slice(None, 1), slice(1, None)]},
        {'shape': (2,), 'axis': 0, 'ys_section': [],
         'slices': [slice(None, None)]},
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestSplitAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.arange(
            numpy.prod(self.shape), dtype=self.dtype).reshape(self.shape)
        self.ys = [self.x[s] for s in self.slices]

    def check_forward(self, x_data, ys_data, indices_or_sections, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(
            x, indices_or_sections, axis, force_tuple=True)
        for yd, y in zip(ys_data, ys):
            self.assertEqual(y.data.dtype, self.dtype)
            self.assertIsInstance(y.data.shape, tuple)
            testing.assert_allclose(yd, y.data, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.ys, self.ys_section, self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x),
            [cuda.to_gpu(y.copy()) for y in self.ys],
            self.ys_section, axis=self.axis)

    def check_backward(self, x_data, indices_or_sections, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(
            x, indices_or_sections, axis, force_tuple=True)
        for y in ys:
            y.grad = y.data
        ys[0].backward()

        testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.ys_section, axis=self.axis)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.ys_section, axis=self.axis)


class TestSplitAxisNone(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array([1, 2], dtype=numpy.float32)
        self.ys_section = [1]
        self.axis = 0

    def check_backward(self, x_data, indices_or_sections, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(x, indices_or_sections, axis)
        # Only set ys[0]
        ys[0].grad = ys[0].data
        ys[0].backward()

        gx = numpy.array([1, 0])
        testing.assert_allclose(gx, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.ys_section, axis=self.axis)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.ys_section, axis=self.axis)


class TestSplitAxisForceArray(unittest.TestCase):

    def setUp(self):
        self.x = numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)
        self.axis = 1

    def check_forward_force_tuple(self, x_data, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(x, 1, axis, force_tuple=True)
        self.assertIsInstance(ys, tuple)
        self.assertEqual(len(ys), 1)

    def test_forward_force_tuple_cpu(self):
        self.check_forward_force_tuple(self.x, self.axis)

    @attr.gpu
    def test_forward_force_tuple_gpu(self):
        self.check_forward_force_tuple(cuda.to_gpu(self.x), axis=self.axis)

    def check_forward_single(self, x_data, axis):
        x = chainer.Variable(x_data)
        ys = functions.split_axis(x, 1, axis)
        self.assertIsInstance(ys, chainer.Variable)

    def test_forward_single_cpu(self):
        self.check_forward_single(self.x, self.axis)

    @attr.gpu
    def test_forward_single_gpu(self):
        self.check_forward_single(cuda.to_gpu(self.x), axis=self.axis)


testing.run_module(__name__, __file__)

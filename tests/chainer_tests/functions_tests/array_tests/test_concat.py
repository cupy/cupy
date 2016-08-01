import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (7, 3), 'axis': 0,
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2,), 'axis': 0, 'slices': [slice(None, 1), slice(1, None)]},
        {'shape': (2,), 'axis': 0, 'slices': [()]},
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
        {'shape': (2, 7, 3), 'axis': -2,
         'slices': [[slice(None), slice(None, 2)], [slice(None), slice(2, 5)],
                    [slice(None), slice(5, None)]]},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestConcat(unittest.TestCase):

    def setUp(self):
        self.y = numpy.arange(
            numpy.prod(self.shape), dtype=self.dtype).reshape(self.shape)
        self.xs = [self.y[s] for s in self.slices]

    def check_forward(self, xs_data, y_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        testing.assert_allclose(y_data, y.data, atol=0, rtol=0)
        self.assertIsInstance(y.data.shape, tuple)

    def test_forward_cpu(self):
        self.check_forward(self.xs, self.y, axis=self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            [cuda.to_gpu(x.copy()) for x in self.xs],
            cuda.to_gpu(self.y), axis=self.axis)

    def check_backward(self, xs_data, axis):
        xs = tuple(chainer.Variable(x_data) for x_data in xs_data)
        y = functions.concat(xs, axis=axis)
        y.grad = y.data
        y.backward()

        for x in xs:
            testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.xs, axis=self.axis)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward([cuda.to_gpu(x.copy()) for x in self.xs],
                            axis=self.axis)


class TestConcatInvalidAxisType(unittest.TestCase):

    def test_invlaid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Concat('a')


testing.run_module(__name__, __file__)

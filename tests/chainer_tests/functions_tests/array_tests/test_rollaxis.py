import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'axis': 0, 'start': 2, 'out_shape': (3, 2, 4)},
    {'axis': 2, 'start': 0, 'out_shape': (4, 2, 3)},
    {'axis': 1, 'start': 1, 'out_shape': (2, 3, 4)},
    {'axis': -3, 'start': 2, 'out_shape': (3, 2, 4)},
    {'axis': -1, 'start': 0, 'out_shape': (4, 2, 3)},
    {'axis': -2, 'start': -2, 'out_shape': (2, 3, 4)},
)
class TestRollaxis(unittest.TestCase):

    dtype = numpy.float32

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.out_shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.rollaxis(x, self.axis, self.start)

        expect = numpy.rollaxis(self.x, self.axis, self.start)
        testing.assert_allclose(y.data, expect)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        gradient_check.check_backward(
            functions.Rollaxis(self.axis, self.start), x_data, g_data)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

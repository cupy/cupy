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
        {'shape': (3, 4)},
        {'shape': ()},
    ],
    [
        {'in_type': numpy.float16},
        {'in_type': numpy.float32},
        {'in_type': numpy.float64},
    ],
    [
        {'out_type': numpy.float16},
        {'out_type': numpy.float32},
        {'out_type': numpy.float64},
    ]
))
class TestCast(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.in_type)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.out_type)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.cast(x, self.out_type)
        self.assertEqual(y.data.shape, x.data.shape)
        self.assertEqual(y.data.dtype, self.out_type)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        func = functions.Cast(self.out_type)
        gradient_check.check_backward(
            func, x_data, g_data, eps=2.0 ** -2, atol=1e-3, rtol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)


testing.run_module(__name__, __file__)

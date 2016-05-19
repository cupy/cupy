import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'shape': (3,), 'dtype': 'f', 'axis': 0, 'rev': False},
    {'shape': (3,), 'dtype': 'f', 'axis': -1, 'rev': True},
    {'shape': (3, 4), 'dtype': 'd', 'axis': 1, 'rev': True},
    {'shape': (3, 4, 5), 'dtype': 'f', 'axis': 2, 'rev': False},
)
class TestPermutate(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.indices = numpy.random.permutation(self.shape[self.axis])

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.permutate(x, self.indices, axis=self.axis, rev=self.rev)

        y_cpu = cuda.to_cpu(y.data)
        y_cpu = numpy.rollaxis(y_cpu, axis=self.axis)
        x_data = numpy.rollaxis(self.x, axis=self.axis)
        for i, ind in enumerate(self.indices):
            if self.rev:
                numpy.testing.assert_array_equal(y_cpu[ind], x_data[i])
            else:
                numpy.testing.assert_array_equal(y_cpu[i], x_data[ind])

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        fun = functions.Permutate(self.indices, axis=self.axis, rev=self.rev)
        gradient_check.check_backward(
            fun, x_data, g_data)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))

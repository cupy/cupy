import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'shape': (3,), 'dtype': 'f', 'axis': 0, 'inv': False},
    {'shape': (3,), 'dtype': 'f', 'axis': -1, 'inv': True},
    {'shape': (3, 4), 'dtype': 'd', 'axis': 1, 'inv': True},
    {'shape': (3, 4, 5), 'dtype': 'f', 'axis': 2, 'inv': False},
)
class TestPermutate(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.indices = numpy.random.permutation(
            self.shape[self.axis]).astype(numpy.int32)

    def check_forward(self, x_data, ind_data):
        x = chainer.Variable(x_data)
        indices = chainer.Variable(ind_data)
        y = functions.permutate(x, indices, axis=self.axis, inv=self.inv)

        y_cpu = cuda.to_cpu(y.data)
        y_cpu = numpy.rollaxis(y_cpu, axis=self.axis)
        x_data = numpy.rollaxis(self.x, axis=self.axis)
        for i, ind in enumerate(self.indices):
            if self.inv:
                numpy.testing.assert_array_equal(y_cpu[ind], x_data[i])
            else:
                numpy.testing.assert_array_equal(y_cpu[i], x_data[ind])

    def test_forward_cpu(self):
        self.check_forward(self.x, self.indices)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.indices))

    def check_backward(self, x_data, ind_data, g_data):
        fun = functions.Permutate(axis=self.axis, inv=self.inv)
        gradient_check.check_backward(
            fun, (x_data, ind_data), g_data)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.indices, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.indices),
                            cuda.to_gpu(self.g))


@testing.parameterize(
    {'indices': [0, 0]},
    {'indices': [-1, 0]},
    {'indices': [0, 2]},
)
class TestPermutateInvalidIndices(unittest.TestCase):

    def setUp(self):
        self.x = numpy.arange(10).reshape((2, 5)).astype('f')
        self.ind = numpy.array(self.indices, 'i')
        self.debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.debug)

    def check_invalid(self, x_data, ind_data):
        x = chainer.Variable(x_data)
        ind = chainer.Variable(ind_data)
        with self.assertRaises(ValueError):
            functions.permutate(x, ind)

    def test_invlaid_cpu(self):
        self.check_invalid(self.x, self.ind)

    @attr.gpu
    def test_invlaid_gpu(self):
        self.check_invalid(cuda.to_gpu(self.x), cuda.to_gpu(self.ind))


testing.run_module(__name__, __file__)

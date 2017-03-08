import unittest

import itertools
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(
    {'shape': (4, 15), 'axis': 1},
    {'shape': (4, 3, 2, 5), 'axis': 0},
    {'shape': (4, 3, 2, 5), 'axis': 1},
    {'shape': (4, 3, 2, 5), 'axis': 2},
    {'shape': (4, 3, 2, 5), 'axis': 3},
)
class TestL2Normalization(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x_data, axis):
        x = chainer.Variable(x_data)

        y = functions.normalize(x, axis=axis)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.empty_like(self.x)
        shape = self.x.shape
        indices = []
        for i in six.moves.range(len(shape)):
            if i != axis:
                indices.append(six.moves.range(shape[i]))
            else:
                indices.append([slice(None)])
        indices_tuple = list(itertools.product(*indices))
        for index in indices_tuple:
            y_expect[index] = self.x[index] / numpy.linalg.norm(self.x[index])
        testing.assert_allclose(y_expect, y_data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        gradient_check.check_backward(
            functions.NormalizeL2(axis=axis), x_data, y_grad,
            rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.axis, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy))

    def check_eps(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.normalize(x, axis=self.axis)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.zeros_like(self.x)
        testing.assert_allclose(y_expect, y_data)

    def test_eps_cpu(self):
        self.check_eps(numpy.zeros_like(self.x))

    @attr.gpu
    def test_eps_gpu(self):
        self.check_eps(cuda.to_gpu(numpy.zeros_like(self.x)))


testing.run_module(__name__, __file__)

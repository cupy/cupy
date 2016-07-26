import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(
    {'shape': (4, 15)},
)
class TestL2Normalization(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.normalize(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.empty_like(self.x)
        for n in six.moves.range(len(self.x)):
            y_expect[n] = self.x[n] / numpy.linalg.norm(self.x[n])

        testing.assert_allclose(y_expect, y_data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.NormalizeL2(), x_data, y_grad, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_eps(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.normalize(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.zeros_like(self.x)
        testing.assert_allclose(y_expect, y_data)

    def test_eps_cpu(self):
        self.check_eps(numpy.zeros_like(self.x))

    @attr.gpu
    def test_eps_gpu(self):
        self.check_eps(cuda.to_gpu(numpy.zeros_like(self.x)))


class TestL2NormalizationTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x = chainer.Variable(numpy.zeros((4, 3, 24), dtype=numpy.float32))

        with self.assertRaises(type_check.InvalidType):
            chainer.functions.normalize(x)


testing.run_module(__name__, __file__)

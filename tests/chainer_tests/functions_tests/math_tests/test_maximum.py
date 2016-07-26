import unittest

import chainer
import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestMaximum(unittest.TestCase):

    def setUp(self):
        shape = self.shape
        self.x1 = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        # Avoid close values for stability in numerical gradient.
        for i in numpy.ndindex(shape):
            if -0.125 < self.x1[i] - self.x2[i] < 0.125:
                self.x1[i] = -0.5
                self.x2[i] = 0.5
        self.y_expected = numpy.maximum(self.x1, self.x2)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x1_data, x2_data, y_expected):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.maximum(x1, x2)
        self.assertEqual(y.data.dtype, self.dtype)
        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2, self.y_expected)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        self.check_forward(x1, x2, self.y_expected)

    def check_backward(self, x1_data, x2_data, y_grad):
        func = functions.maximum
        x = (x1_data, x2_data)
        gradient_check.check_backward(
            func, x, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x1, x2, gy)


@testing.parameterize(*testing.product({
    'dtype1': [numpy.float16, numpy.float32, numpy.float64],
    'dtype2': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestMaximumInconsistentTypes(unittest.TestCase):

    def test_maximum_inconsistent_types(self):
        if self.dtype1 == self.dtype2:
            return
        x1_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype1)
        x2_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype2)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        with self.assertRaises(type_check.InvalidType):
            functions.maximum(x1, x2)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestMaximumInconsistentShapes(unittest.TestCase):

    def test_maximum_inconsistent_shapes(self):
        x1_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)
        x2_data = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        with self.assertRaises(type_check.InvalidType):
            functions.maximum(x1, x2)

testing.run_module(__name__, __file__)

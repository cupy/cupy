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


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 2, 4)],
        'axis': [None, 0, 1, 2, -1],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'use_weights': [True, False],
        'keepdims': [True, False],
    }) +
    testing.product({
        'shape': [()],
        'axis': [None],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'use_weights': [True, False],
        'keepdims': [True, False],
    })))
class TestAverage(unittest.TestCase):

    def setUp(self):
        ndim = len(self.shape)
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.axis is None:
            w_shape = self.shape
        else:
            axis = self.axis
            if axis < 0:
                axis += ndim
            w_shape = self.shape[axis],

        g_shape = self.x.sum(axis=self.axis, keepdims=self.keepdims).shape
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(self.dtype)
        self.w = numpy.random.uniform(-1, 1, w_shape).astype(self.dtype)

    def check_forward(self, x_data, axis, weights):
        x = chainer.Variable(x_data)
        if self.use_weights:
            w = chainer.Variable(weights)
            w_data = self.w
        else:
            w = None
            w_data = None
        y = functions.average(x, axis=axis, weights=w, keepdims=self.keepdims)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = numpy.average(
            self.x, axis=axis, weights=w_data)
        if self.keepdims:
            # numpy.average does not support keepdims
            if axis is None:
                axis = list(six.moves.range(x_data.ndim))
            elif isinstance(axis, int):
                axis = axis,
            shape = list(x_data.shape)
            for i in six.moves.range(len(shape)):
                if i in axis or i - len(shape) in axis:
                    shape[i] = 1
            y_expect = y_expect.reshape(shape)

        if self.dtype == numpy.float16:
            options = {'atol': 1e-3, 'rtol': 1e-3}
        else:
            options = {}

        self.assertEqual(y_expect.shape, y.shape)
        testing.assert_allclose(y_expect, y.data, **options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis, self.w)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.w))

    def check_backward(self, x_data, y_grad, axis, w_data):
        if self.use_weights:
            def f(x, w):
                return functions.average(
                    x, axis=axis, weights=w, keepdims=self.keepdims)
            args = (x_data, w_data)
        else:
            def f(x):
                return functions.average(x, axis=axis, keepdims=self.keepdims)
            args = x_data

        gradient_check.check_backward(
            f, args, y_grad, atol=1e-2, rtol=1e-2,
            dtype=numpy.float64)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.axis, self.w)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), self.axis,
            cuda.to_gpu(self.w))


testing.run_module(__name__, __file__)

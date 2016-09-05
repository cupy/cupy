import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(
    {'learn_W': True, 'bias_term': False, 'bias_shape': None},
    {'learn_W': True, 'bias_term': True, 'bias_shape': None},
    {'learn_W': False, 'bias_term': False, 'bias_shape': None},
    {'learn_W': False, 'bias_term': True, 'bias_shape': (2,)}
)
class TestScale(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        self.W = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        self.b = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        self.y_expected = numpy.copy(self.x)
        for i, j, k in numpy.ndindex(self.y_expected.shape):
            self.y_expected[i, j, k] *= self.W[j]
            if self.bias_term:
                self.y_expected[i, j, k] += self.b[j]
        self.gy = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)

        bias_term = self.bias_term
        bias_shape = self.bias_shape
        axis = 1
        if self.learn_W:
            self.link = links.Scale(
                axis, self.W.shape, bias_term, bias_shape)
            self.link.W.data = self.W
            if bias_term:
                self.link.bias.b.data = self.b
        else:
            self.link = links.Scale(
                axis, None, bias_term, bias_shape)
            if bias_term:
                self.link.bias.b.data = self.b
        self.link.cleargrads()

    def test_attribute_presence(self):
        self.assertEqual(self.learn_W, hasattr(self.link, 'W'))
        self.assertEqual(self.bias_term, hasattr(self.link, 'bias'))

    def check_forward(self, x_data, W_data, y_expected):
        x = chainer.Variable(x_data)
        if W_data is None:
            y = self.link(x)
            testing.assert_allclose(y_expected, y.data)
        else:
            W = chainer.Variable(W_data)
            y = self.link(x, W)
            testing.assert_allclose(y_expected, y.data)

    def test_forward_cpu(self):
        if self.learn_W:
            W = None
        else:
            W = self.W
        self.check_forward(self.x, W, self.y_expected)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        if self.learn_W:
            W = None
        else:
            W = cuda.to_gpu(self.W)
        self.check_forward(x, W, self.y_expected)

    def check_backward(self, x_data, W_data, y_grad):
        if W_data is None:
            params = [self.link.W]
            gradient_check.check_backward(
                self.link, x_data, y_grad, params, atol=1e-2)
        else:
            gradient_check.check_backward(
                self.link, (x_data, W_data), y_grad, atol=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        if self.learn_W:
            W = None
        else:
            W = self.W
        self.check_backward(self.x, W, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        if self.learn_W:
            W = None
        else:
            W = cuda.to_gpu(self.W)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, W, gy)


class TestScaleInvalidArgc(unittest.TestCase):

    def setUp(self):
        x_data = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        W_data = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        self.axis = 1
        self.x = chainer.Variable(x_data)
        self.W = chainer.Variable(W_data)

    def test_scale_invalid_argc1(self):
        func = links.Scale(self.axis, self.W.data.shape)
        with chainer.DebugMode(True):
            with self.assertRaises(AssertionError):
                func(self.x, self.W)

    def test_scale_invalid_argc2(self):
        func = links.Scale(self.axis, None)
        with chainer.DebugMode(True):
            with self.assertRaises(AssertionError):
                func(self.x)


class TestScaleNoBiasShape(unittest.TestCase):

    def test_scale_no_bias_shape(self):
        axis = 1
        with self.assertRaises(ValueError):
            links.Scale(axis, None, True, None)


testing.run_module(__name__, __file__)

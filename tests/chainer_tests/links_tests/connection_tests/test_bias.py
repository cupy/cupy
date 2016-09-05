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
    {'learn_b': True},
    {'learn_b': False}
)
class TestBias(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        self.b = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        self.y_expected = numpy.copy(self.x)
        for i, j, k in numpy.ndindex(self.y_expected.shape):
            self.y_expected[i, j, k] += self.b[j]
        self.gy = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)

        axis = 1
        if self.learn_b:
            self.link = links.Bias(axis, self.b.shape)
            self.link.b.data = self.b
        else:
            self.link = links.Bias(axis, None)
        self.link.cleargrads()

    def test_attribute_presence(self):
        self.assertEqual(self.learn_b, hasattr(self.link, 'b'))

    def check_forward(self, x_data, b_data, y_expected):
        x = chainer.Variable(x_data)
        if b_data is None:
            y = self.link(x)
            testing.assert_allclose(y_expected, y.data)
        else:
            b = chainer.Variable(b_data)
            y = self.link(x, b)
            testing.assert_allclose(y_expected, y.data)

    def test_forward_cpu(self):
        if self.learn_b:
            b = None
        else:
            b = self.b
        self.check_forward(self.x, b, self.y_expected)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        if self.learn_b:
            b = None
        else:
            b = cuda.to_gpu(self.b)
        self.check_forward(x, b, self.y_expected)

    def check_backward(self, x_data, b_data, y_grad):
        if b_data is None:
            params = [self.link.b]
            gradient_check.check_backward(
                self.link, x_data, y_grad, params, atol=1e-2)
        else:
            gradient_check.check_backward(
                self.link, (x_data, b_data), y_grad, atol=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        if self.learn_b:
            b = None
        else:
            b = self.b
        self.check_backward(self.x, b, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        if self.learn_b:
            b = None
        else:
            b = cuda.to_gpu(self.b)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, b, gy)


class TestBiasInvalidArgc(unittest.TestCase):

    def setUp(self):
        x_data = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        b_data = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        self.axis = 1
        self.x = chainer.Variable(x_data)
        self.b = chainer.Variable(b_data)

    def test_bias_invalid_argc1(self):
        func = links.Bias(self.axis, self.b.data.shape)
        with chainer.DebugMode(True):
            with self.assertRaises(AssertionError):
                func(self.x, self.b)

    def test_bias_invalid_argc2(self):
        func = links.Bias(self.axis, None)
        with chainer.DebugMode(True):
            with self.assertRaises(AssertionError):
                func(self.x)


testing.run_module(__name__, __file__)

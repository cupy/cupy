import os
import tempfile
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'in_shape': [(3,), (3, 2, 2)],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestDropconnect(unittest.TestCase):

    out_size = 2

    def setUp(self):
        in_size = numpy.prod(self.in_shape)
        self.link = links.Dropconnect(
            in_size, self.out_size,
            initialW=chainer.initializers.Normal(1, self.W_dtype),
            initial_bias=chainer.initializers.Normal(1, self.x_dtype))
        W = self.link.W.data
        b = self.link.b.data
        self.link.cleargrads()

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(self.x_dtype)
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 5e-2}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, self.x_dtype)
##        y_expect = self.x.reshape(4, -1).dot(self.link.W.data.T * self.link.mask.T) + self.link.b.data
##        testing.assert_allclose(self.y, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

##    @attr.gpu
##    @condition.retry(3)
##    def test_forward_gpu(self):
##        self.link.to_gpu()
##        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=2 ** -3,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

##    @attr.gpu
##    @condition.retry(3)
##    def test_backward_gpu(self):
##        self.link.to_gpu()
##        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestInvalidDropconnect(unittest.TestCase):

    def setUp(self):
        self.link = links.Dropconnect(3, 2)
        self.x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(self.x))

unittest.main()
#testing.run_module(__name__, __file__)

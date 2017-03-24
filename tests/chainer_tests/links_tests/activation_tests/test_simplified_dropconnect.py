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


def gen_mask(ratio, shape):
    return numpy.random.rand(*shape) >= ratio * (1. / (1 - ratio))


@testing.parameterize(*testing.product({
    'in_shape': [(3,), (3, 2, 2)],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSimplifiedDropconnect(unittest.TestCase):

    out_size = 2
    ratio = 0.5

    def setUp(self):
        in_size = numpy.prod(self.in_shape)

        self.link = links.SimplifiedDropconnect(
            in_size, self.out_size,
            initialW=chainer.initializers.Normal(1, self.W_dtype),
            initial_bias=chainer.initializers.Normal(1, self.x_dtype))
        self.link.cleargrads()
        self.mask = gen_mask(self.ratio, self.link.W.shape)

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(self.x_dtype)
        W = self.link.W.data
        b = self.link.b.data
        self.y_expect = self.x.reshape(4, -1).dot(W.T * self.mask.T) + b
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 5e-2}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, mask):
        x = chainer.Variable(x_data)
        y = self.link(x, True, mask)
        self.assertEqual(y.data.dtype, self.x_dtype)
        testing.assert_allclose(self.y_expect, y.data,
                                **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.mask)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.mask))

    def link_wrapper(self, *data):
        return self.link(data[0], True, data[1])

    def check_backward(self, x_data, y_grad, mask):
        gradient_check.check_backward(
            self.link_wrapper, (x_data, mask), y_grad,
            (self.link.W, self.link.b), eps=2 ** -3,
            no_grads=(False, True), **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.mask)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                            cuda.to_gpu(self.mask))


class TestSimplifiedDropconnectParameterShapePlaceholder(unittest.TestCase):

    in_size = 3
    in_shape = (in_size,)
    out_size = 2
    in_size_or_none = None
    ratio = 0.5

    def setUp(self):
        self.link = links.SimplifiedDropconnect(self.in_size_or_none,
                                                self.out_size)
        temp_x = numpy.random.uniform(-1, 1,
                                      (self.out_size,
                                       self.in_size)).astype(numpy.float32)
        self.link(chainer.Variable(temp_x))
        W = self.link.W.data
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()
        self.mask = gen_mask(self.ratio, self.link.W.shape)

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y_expect = self.x.reshape(4, -1).dot(W.T * self.mask.T) + b

    def check_forward(self, x_data, mask):
        x = chainer.Variable(x_data)
        y = self.link(x, True, mask)
        self.assertEqual(y.data.dtype, numpy.float32)
        testing.assert_allclose(self.y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.mask)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.mask))

    def link_wrapper(self, *data):
        return self.link(data[0], True, data[1])

    def check_backward(self, x_data, y_grad, mask):
        gradient_check.check_backward(
            self.link_wrapper, (x_data, mask), y_grad,
            (self.link.W, self.link.b), eps=1e-2, no_grads=(False, True))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.mask)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                            cuda.to_gpu(self.mask))

    def test_serialization(self):
        lin1 = links.SimplifiedDropconnect(None, self.out_size)
        x = chainer.Variable(self.x)
        # Must call the link to initialize weights.
        lin1(x)
        w1 = lin1.W.data
        fd, temp_file_path = tempfile.mkstemp()
        os.close(fd)
        npz.save_npz(temp_file_path, lin1)
        lin2 = links.Dropconnect(None, self.out_size)
        npz.load_npz(temp_file_path, lin2)
        w2 = lin2.W.data
        self.assertEqual((w1 == w2).all(), True)


class TestInvalidSimplifiedDropconnect(unittest.TestCase):

    def setUp(self):
        self.link = links.SimplifiedDropconnect(3, 2)
        self.x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(self.x))


testing.run_module(__name__, __file__)

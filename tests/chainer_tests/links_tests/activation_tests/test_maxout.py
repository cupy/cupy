import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


@testing.parameterize(
    *testing.product(
        {'in_shape': [(2, ), (2, 5)],
         'num_channel': [3],
         'out_size': [4],
         'wscale': [1],
         'initial_bias': ['random', None],
         'batchsize': [7]}
    )
)
class TestMaxout(unittest.TestCase):

    def setUp(self):
        # x, W, and b are set so that the result of forward
        # propagation gets stable, meaning that their small pertubations
        # do not change :math:`argmax_{j} W_{\cdot ij} x + b_{ij}`.

        x_shape = (self.batchsize, ) + self.in_shape
        self.x = numpy.random.uniform(
            -0.05, 0.05, x_shape).astype(numpy.float32) + 1
        self.gy = numpy.random.uniform(
            -0.05, 0.05, (self.batchsize, self.out_size)
        ).astype(numpy.float32)

        in_size = numpy.prod(self.in_shape)
        initialW = numpy.random.uniform(
            -0.05, 0.05, (in_size, self.num_channel, self.out_size))
        for c in six.moves.range(self.num_channel):
            w = numpy.arange(in_size, dtype=numpy.float32) + 1
            for o in six.moves.range(self.out_size):
                initialW[:, c, o] += w * o

        initial_bias = None
        if self.initial_bias == 'random':
            initial_bias = numpy.random.uniform(
                -0.05, 0.05, (self.num_channel, self.out_size))

        self.link = links.Maxout(in_size, self.num_channel, self.out_size,
                                 self.wscale, initialW, initial_bias)

        self.W = self.link.W.data.copy()
        self.y = numpy.tensordot(_as_mat(self.x), self.W, axes=1)
        if self.initial_bias is not None:
            self.b = self.link.b.data.copy()
            self.y += self.b
        self.y = numpy.max(self.y, axis=1)
        self.link.zerograds()

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.link(x)
        y.grad = y_grad
        y.backward()

        f = lambda: (self.link(x).data, )
        if self.initial_bias is None:
            gx, gW = gradient_check.numerical_grad(
                f, (x.data, self.link.W.data),
                (y.grad, ), eps=1e-4)
        else:
            gx, gW, gb = gradient_check.numerical_grad(
                f, (x.data, self.link.W.data, self.link.b.data),
                (y.grad, ), eps=1e-4)

        gradient_check.assert_allclose(gx, x.grad, atol=1e-2)
        gradient_check.assert_allclose(gW, self.link.W.grad, atol=1e-2)
        if self.initial_bias is not None:
            gradient_check.assert_allclose(gb, self.link.b.grad, atol=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestInvalidMaxout(unittest.TestCase):

    def setUp(self):
        self.link = links.Maxout(2, 3, 4)
        self.x = numpy.random.uniform(
            -1, 1, (10, 7)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(self.x))


class TestInitialization(unittest.TestCase):

    def setUp(self):
        self.initialW = numpy.random.uniform(
            -1, 1, (2, 3, 4)).astype(numpy.float32)
        self.initial_bias = numpy.random.uniform(
            -1, 1, (3, 4)).astype(numpy.float32)
        self.link = links.Maxout(
            2, 3, 4, initialW=self.initialW,
            initial_bias=self.initial_bias)

    def check_param(self):
        gradient_check.assert_allclose(self.initialW, self.link.W.data)
        gradient_check.assert_allclose(self.initial_bias, self.link.b.data)

    def test_param_cpu(self):
        self.check_param()

    @attr.gpu
    def test_param_gpu(self):
        self.link.to_gpu()
        self.check_param()


testing.run_module(__name__, __file__)

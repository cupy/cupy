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


def _maxout(x, W, b):
    W_r = numpy.rollaxis(W, 2)
    y = numpy.tensordot(_as_mat(x), W_r, axes=1)
    if b is not None:
        y += b
    return numpy.max(y, axis=2)


@testing.parameterize(
    *testing.product(
        {'in_shape': [(2, ), (2, 5)],
         'pool_size': [3],
         'out_size': [4],
         'wscale': [1],
         'initial_bias': ['random', 'scalar', None],
         'batchsize': [7]}
    )
)
class TestMaxout(unittest.TestCase):

    def setUp(self):
        # x, W, and b are set so that the result of forward
        # propagation gets stable, meaning that their small pertubations
        # do not change :math:`argmax_{j} W_{ij\cdot} x + b_{ij}`.

        x_shape = (self.batchsize, ) + self.in_shape
        self.x = numpy.random.uniform(
            -0.05, 0.05, x_shape).astype(numpy.float32) + 1
        self.gy = numpy.random.uniform(
            -0.05, 0.05, (self.batchsize, self.out_size)
        ).astype(numpy.float32)

        in_size = numpy.prod(self.in_shape)
        initialW = numpy.random.uniform(
            -0.05, 0.05, (self.out_size, self.pool_size, in_size)
        ).astype(numpy.float32)
        for o in six.moves.range(self.out_size):
            w = numpy.arange(in_size, dtype=numpy.float32) + 1
            for c in six.moves.range(self.pool_size):
                initialW[o, c, :] += w * c

        if self.initial_bias == 'random':
            initial_bias = numpy.random.uniform(
                -0.05, 0.05, (self.out_size, self.pool_size))
        elif self.initial_bias == 'scalar':
            initial_bias = numpy.full(
                (self.out_size, self.pool_size), 5, dtype=numpy.float32)
        elif self.initial_bias is None:
            initial_bias = None

        self.link = links.Maxout(in_size, self.out_size, self.pool_size,
                                 self.wscale, initialW, initial_bias)

        self.y = _maxout(self.x, initialW, initial_bias)
        self.link.cleargrads()

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        testing.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        params = [self.link.linear.W]
        if self.initial_bias is not None:
            params.append(self.link.linear.b)
        gradient_check.check_backward(
            self.link, x_data, y_grad, params, atol=1e-2)

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
        self.in_size = 2
        self.out_size = 3
        self.pool_size = 4
        self.initialW = numpy.random.uniform(
            -1, 1, (self.out_size, self.pool_size, self.in_size)
        ).astype(numpy.float32)
        self.initial_bias = numpy.random.uniform(
            -1, 1, (self.out_size, self.pool_size)
        ).astype(numpy.float32)
        self.link = links.Maxout(
            self.in_size, self.out_size, self.pool_size,
            initialW=self.initialW, initial_bias=self.initial_bias)

    def check_param(self):
        linear_out_size = self.out_size * self.pool_size
        initialW = self.initialW.reshape((linear_out_size, -1))
        testing.assert_allclose(initialW, self.link.linear.W.data)
        initial_bias = self.initial_bias.reshape((linear_out_size,))
        testing.assert_allclose(initial_bias, self.link.linear.b.data)

    def test_param_cpu(self):
        self.check_param()

    @attr.gpu
    def test_param_gpu(self):
        self.link.to_gpu()
        self.check_param()


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.testing import type_check


@testing.parameterize(
    {'in_shape': (2, ), 'num_channel': 3, 'out_size': 4, initial_bias: 0, 'batchsize': 7}
    {'in_shape': (2, 5), 'num_channel': 3, 'out_size': 4, initial_bias: None, 'batchsize': 7}
)
class TestLinear(unittest.Testing):

    def setUp(self):
        in_size = numpy.prod(self.in_shape)
        self.link = links.Maxout(in_size, self.num_channel,
                                 self.out_size, self.initial_bias)
        W = self.link.W.data
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        if self.initial_bias is not None:
            b = self.link.b.data
            b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.zerograds()

        self.W = W.copy()
        if self.initial_bias is not None:
            self.b = b.copy()

        x_shape = (self.batchsize, ) + self.in_shape
        self.x = numpy.random.uniform(
            -1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (self.batchsize, self.out_size).astype(numpy.float32))

        def check_foward(self, x_data):
            x = chainer.Variable(x_data)
            y = self.link(x)
            self.assertEqual(y.data.dtype, numpy.float32)
            gradient_check.assert_allclose(self.y, y.data)

        @condition.retry(3)
        def test_forward_cpu(self):
            self.check_forward(self.x)

        @attr.gpu
        @condition.retry(3)
        def test_forwrad_gpu(self):
            self.link.to_gpu()
            self.check_forward(cuda.to_gpu(self.x))
        
        def check_backward(self, x_data, y_grad):
            x = chainer.Variable(x_data)
            y = self.link(x)
            y.grad = y_grad
            y.backward()

            f = lambda: (self.link(x).data, )
            if initial_bias is None:
                gx, gW = gradient_check.numerical_grad(
                    f, (x.data, self.link.W.data),
                    (y.grad, ), eps=1e-2)
            else:
            gx, gW, gb = gradient_check.numerical_grad(
                f, (x.data, self.link.W.data, self.link.b.data),
                (y.grad, ), eps=1e-2)

            gradient_check.assert_allclose(gx, x.grad)
            gradient_check.assert_allclose(gW, self.link.W.grad)
            if initial_bias is None:
                gradient_check.assert_allclose(gb, self.link.b.grad)

        @condition.retry(3)
        def test_backward_cpu(self):
            self.check_backward(self.x, self.gy)

        @attr.gpu
        @attr.condition.retry(3)
        def test_backward_gpu(self):
            self.link.to_gpu()
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestInvalidMaxout(unittest.TestCase):

    def setUp(self):
        self.link = links.Maxout(2, 3, 4):
        self.x = numpy.random.uniform(
            -1, 1, (10, 7)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(self.x))


testing.run_module(__name__, __file__)

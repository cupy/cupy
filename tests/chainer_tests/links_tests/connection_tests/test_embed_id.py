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
    {'x_data': [0, 1, 0]},
    {'x_data': [[0, 1, 0], [1, 0, 1]]},
)
class TestEmbedID(unittest.TestCase):

    def setUp(self):
        self.link = links.EmbedID(3, 2)
        self.link.zerograds()

        self.W = self.link.W.data.copy()  # fixed on CPU
        self.x = numpy.array(self.x_data, dtype=numpy.int32)
        y_shape = self.x.shape + (2,)
        self.gy = numpy.random.uniform(-1, 1, y_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.empty_like(self.gy)
        for i in numpy.ndindex(self.x.shape):
            y_expect[i] = self.W[int(self.x[i])]

        gradient_check.assert_allclose(y_expect, y.data, atol=0, rtol=0)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, self.link.W)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

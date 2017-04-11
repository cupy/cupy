import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestDepthwiseConvolution2D(unittest.TestCase):

    def setUp(self):
        self.link = links.DepthwiseConvolution2D(
            3, 2, 3, stride=2, pad=1,
            initialW=chainer.initializers.Normal(1, self.W_dtype),
            initial_bias=chainer.initializers.Normal(1, self.x_dtype))
        self.link.cleargrads()

        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 6, 2, 2)).astype(self.x_dtype)
        self.check_backward_options = {}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_backward_options = {'atol': 3e-2, 'rtol': 5e-2}

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=2 ** -3,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestDepthWiseConvolution2DParameterShapePlaceholder(unittest.TestCase):

    def setUp(self):
        in_channels = None
        self.link = links.DepthwiseConvolution2D(in_channels, 2, 3,
                                                 stride=2, pad=1)
        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(numpy.float32)
        self.link(chainer.Variable(self.x))
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 6, 2, 2)).astype(numpy.float32)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

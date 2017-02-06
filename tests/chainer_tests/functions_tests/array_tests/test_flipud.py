import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3,), (3, 4)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestFlipUD(unittest.TestCase):

    shape = (3, 4)
    dtype = numpy.float32

    def setUp(self):
        self.x = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.flipud(x)

        testing.assert_allclose(y.data, numpy.flipud(self.x))

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.FlipUD(), x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

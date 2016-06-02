import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'in_shape': [(4, 3, 2)],
    'axes': [(-1, 0, 1), None],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
class TestTranspose(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)

    def check_forward(self, x_data):
        axes = self.axes
        x = chainer.Variable(x_data)
        y = functions.transpose(x, axes)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertTrue((self.x.transpose(axes) == cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.transpose(x, self.axes)
        y.grad = y.data
        y.backward()
        testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.x)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(3, 2, 4)],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
class TestWhere(unittest.TestCase):

    def setUp(self):
        self.c_data = numpy.random.uniform(-1, 1, self.shape) > 0
        self.x_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.y_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.g_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_backward_options = {'dtype': numpy.float64}

    def check_forward(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        z = functions.where(c, x, y)

        self.assertEqual(x.data.shape, z.data.shape)

        for i in numpy.ndindex(c.data.shape):
            if c.data[i]:
                self.assertEqual(x.data[i], z.data[i])
            else:
                self.assertEqual(y.data[i], z.data[i])

    def test_forward_cpu(self):
        self.check_forward(self.c_data, self.x_data, self.y_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_data),
                           cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.y_data))

    def check_backward(self, c_data, x_data, y_data, g_data):
        gradient_check.check_backward(
            functions.Where(), (c_data, x_data, y_data), g_data,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.c_data, self.x_data, self.y_data, self.g_data)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.c_data),
                            cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.y_data),
                            cuda.to_gpu(self.g_data))


testing.run_module(__name__, __file__)

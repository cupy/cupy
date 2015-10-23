import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.testing import attr


class TestWhere(unittest.TestCase):

    shape = (3, 2, 4)

    def setUp(self):
        self.c_data = numpy.random.uniform(-1, 1, self.shape) > 0
        self.x_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.y_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        z = F.where(c, x, y)

        self.assertEqual(x.data.shape, z.data.shape)

        for c, x, y, z in zip(c.data.flatten(), x.data.flatten(),
                              y.data.flatten(), z.data.flatten()):
            if c:
                self.assertEqual(x, z)
            else:
                self.assertEqual(y, z)

    def test_forward_cpu(self):
        self.check_forward(self.c_data, self.x_data, self.y_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_data),
                           cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.y_data))

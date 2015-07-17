import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing


if cuda.available:
    cuda.init()


class TestDropout(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_type_forward(self, x_data):
        x = chainer.Variable(x_data)
        try:
            functions.dropout(x)
        except Exception:
            self.fail()

    def test_type_forward_cpu(self):
        self.check_type_forward(self.x)

    def test_type_forward_gpu(self):
        self.check_type_forward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
